"""The layer rules: each layer imports only what it is allowed."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
HOST = ROOT / "mizuoto"

NEVER_IN_HOST = {"torch", "torchaudio", "transformers", "onnxruntime"}
MODEL_CODE = frozenset({"mizuoto.runtime", "mizuoto.nn", "mizuoto.engines", "mizuoto.training"})
"""What imports torch somewhere: the core reaches it only inside functions of ``core/inference``."""
STDLIB_ONLY = ("core/codes.py", "core/jobs/events.py", "core/jobs/jsonl.py", "core/remote/frames.py", "core/inference/tasks.py")
"""Shared with the runtime, the engines and job workers: standard library only (and each other)."""
SHARED = {"mizuoto.core.codes", "mizuoto.core.inference.tasks"}
"""The parts of the core the runtime and the engines may import (the error codes and the task types)."""

# layer -> (allowed first-party prefixes, allowed third-party top-level modules)
HOST_RULES: dict[str, tuple[set[str], set[str]]] = {
    "core": (
        {"mizuoto.core", "mizuoto.version", "mizuoto.logger", *MODEL_CODE},
        {"pydantic", "httpx", "numpy", "av", "tomli", "tomli_w", "send2trash", "typing_extensions"},
    ),
    "api": (
        {"mizuoto.api", "mizuoto.core", "mizuoto.version", "mizuoto.logger", "mizuoto.webui"},
        {"fastapi", "starlette", "socketio", "uvicorn", "pydantic", "anyio"},
    ),
    "cli": ({"mizuoto.cli", "mizuoto.core", "mizuoto.api", "mizuoto.version", "mizuoto.logger"}, {"typer", "click", "rich", "pydantic", "uvicorn", "httpx"}),
}

# The native runtime, its building blocks and the engines.
NATIVE_RULES: dict[str, tuple[set[str], set[str]]] = {
    "runtime": ({"mizuoto.runtime", *SHARED}, {"numpy", "torch", "safetensors", "flash_attn", "flash_attn_interface", "xformers", "sageattention", "sageattn3"}),
    "nn": ({"mizuoto.nn", "mizuoto.runtime"}, {"numpy", "torch", "soxr"}),
    "text": ({"mizuoto.text"}, {"tokenizers", "sentencepiece", "tiktoken", "kaldifst", "numpy"}),
    "engines": (
        {"mizuoto.runtime", "mizuoto.nn", "mizuoto.text", "mizuoto.engines", "mizuoto.training", *SHARED},
        {"numpy", "torch", "safetensors", "tokenizers", "sentencepiece", "tiktoken", "soxr", "pypinyin", "rjieba", "yaml", "inflect", "regex", "fugashi"}
        # GPT-SoVITS's text front end (pyopenjtalk: Japanese; jamo, mecab: the Korean extra)
        | {"jieba", "opencc", "wordsegment", "budoux", "fasttext", "ToJyutping", "pyopenjtalk", "jamo", "mecab"},
    ),
    # Training stages run as job workers: the native modules, and the job-worker events.
    "training": (
        {"mizuoto.training", "mizuoto.runtime", "mizuoto.nn", "mizuoto.text", "mizuoto.engines", *SHARED, "mizuoto.core.jobs.events", "mizuoto.core.jobs.jsonl"},
        {"numpy", "torch"},
    ),
}
FORBIDDEN_CALLS = {
    "os.chdir": "changes the working directory",
    "sys.path.append": "changes sys.path",
    "sys.path.insert": "changes sys.path",
    "torch.manual_seed": "seeds the global RNG (use ctx.generator_for)",
    "torch.cuda.manual_seed": "seeds the global RNG",
    "torch.cuda.manual_seed_all": "seeds the global RNG",
    "random.seed": "seeds the global RNG (use ctx.rng)",
    "np.random.seed": "seeds the global RNG",
    "numpy.random.seed": "seeds the global RNG",
    "logging.basicConfig": "configures logging for the whole process",
    "os.putenv": "changes the environment",
    "torch.set_default_dtype": "changes a process-wide default",
    "torch.use_deterministic_algorithms": "changes a process-wide flag",
    "torch.set_float32_matmul_precision": "changes a process-wide flag",
    "sdpa_kernel": "switches SDPA's kernels for every thread (call the aten operator; mizuoto.runtime.accel)",
    "torch.nn.attention.sdpa_kernel": "switches SDPA's kernels for every thread",
}
PROCESS_FLAGS = "mizuoto/runtime/accel/features.py"
"""The one place native code may set torch's process-wide switches (``compute.fast``, ``training.deterministic``), called by
the host and the training stage."""
PROCESS_FLAG_CALLS = {"torch.use_deterministic_algorithms"}
"""The forbidden calls ``PROCESS_FLAGS`` may make."""


def _stdlib() -> set[str]:
    return set(sys.stdlib_module_names) | {"tomllib"}


def _type_checking(node: ast.AST) -> bool:
    return isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"


def _imports(path: Path) -> list[tuple[str, bool]]:
    """(module, imported lazily: inside a function or an ``if TYPE_CHECKING`` block) for every absolute import."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[str, bool]] = []

    def visit(node: ast.AST, nested: bool) -> None:
        for child in ast.iter_child_nodes(node):
            inner = nested or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) or _type_checking(child)
            if isinstance(child, ast.Import):
                out.extend((a.name, nested) for a in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module and child.level == 0:
                # ``from mizuoto.core import codes`` imports the module mizuoto.core.codes
                subs = [f"{child.module}.{a.name}" for a in child.names if _is_module(f"{child.module}.{a.name}")]
                out.extend((name, nested) for name in subs or [child.module])
            visit(child, inner)

    visit(tree, False)
    return out


def _is_module(dotted: str) -> bool:
    parts = dotted.split(".")
    if parts[0] != "mizuoto":
        return False
    path = ROOT.joinpath(*parts)
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def _allowed_first_party(name: str, allowed: set[str]) -> bool:
    return any(name == a or name.startswith(a + ".") for a in allowed)


def _check(root: Path, layer: str, own: set[str], third: set[str], lazy_only: frozenset[str] = frozenset(), lazy_first_party: frozenset[str] = frozenset()) -> list[str]:
    stdlib = _stdlib()
    bad: list[str] = []
    for path in (root / layer).rglob("*.py"):
        rel = path.relative_to(root.parent).as_posix()
        for name, nested in _imports(path):
            top = name.split(".")[0]
            if top == "mizuoto":
                if not _allowed_first_party(name, own):
                    bad.append(f"{rel}: {name}")
                elif not nested and _allowed_first_party(name, set(lazy_first_party)):
                    bad.append(f"{rel}: {name} (only inside functions)")
            elif top in stdlib or top == "__future__":
                continue
            elif top not in third:
                bad.append(f"{rel}: {name}")
            elif top in lazy_only and not nested:
                bad.append(f"{rel}: {name} (only inside functions)")
    return bad


@pytest.mark.parametrize("layer", sorted(HOST_RULES))
def test_host_layer_imports(layer: str) -> None:
    own, third = HOST_RULES[layer]
    bad = _check(HOST, layer, own, third, lazy_first_party=MODEL_CODE if layer == "core" else frozenset())
    assert not bad, "Imports outside the layer rule:\n" + "\n".join(bad)


def test_only_core_inference_reaches_model_code() -> None:
    """``core/inference`` is the core's only door to the runtime, the engines and the trainers."""
    bad = [
        f"{p.relative_to(ROOT)}: {name}"
        for p in (HOST / "core").rglob("*.py")
        if p.parent.name != "inference"
        for name, _ in _imports(p)
        if any(name == m or name.startswith(m + ".") for m in MODEL_CODE)
    ]
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("module", STDLIB_ONLY)
def test_shared_modules_are_standard_library_only(module: str) -> None:
    stdlib = _stdlib()
    shared = {"mizuoto." + m.removesuffix(".py").replace("/", ".") for m in STDLIB_ONLY} | {"mizuoto.core"}
    bad = [name for name, _ in _imports(HOST / module) if name.split(".")[0] not in stdlib | {"__future__"} and name not in shared]
    assert not bad, f"{module} imports {bad}"


@pytest.mark.parametrize("layer", sorted(NATIVE_RULES))
def test_native_layer_imports(layer: str) -> None:
    own, third = NATIVE_RULES[layer]
    bad = _check(HOST, layer, own, third)
    assert not bad, "Imports outside the layer rule:\n" + "\n".join(bad)


NATIVE_PACKAGES = ROOT / "packages" / "native"
"""Engines distributed on their own (licence-gated, decision D1): as ``mizuoto.engines`` but outside the host package.
Each package's own tests hold it to these rules (``check_native_package``), so a change to one is checked there."""


def check_native_package(package: Path) -> None:
    """A gated engine's package (``packages/native/<id>/mizuoto_<id>``) imports like an engine, leaves the process
    state alone and computes attention only through ``mizuoto.runtime.accel``."""
    own, third = NATIVE_RULES["engines"]
    bad = _check(package.parent, package.name, own - {"mizuoto.engines"}, third | {package.name})
    assert not bad, "Imports outside the layer rule:\n" + "\n".join(bad)
    paths = sorted(package.rglob("*.py"))
    bad = _global_state_changes(paths) + _attention_outside_accel(paths)
    assert not bad, "\n".join(bad)


def test_engines_import_no_other_engine() -> None:
    bad = []
    for path in (HOST / "engines").rglob("*.py"):
        parts = path.relative_to(HOST / "engines").parts
        if len(parts) < 2:
            continue
        own = f"mizuoto.engines.{parts[0]}"
        for name, _ in _imports(path):
            if name.startswith("mizuoto.engines.") and not (name == own or name.startswith(own + ".")):
                bad.append(f"{path.relative_to(ROOT)}: {name}")
    assert not bad, "\n".join(bad)


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _global_state_changes(paths: list[Path]) -> list[str]:
    """What in ``paths`` changes process-wide state: engines share the process."""
    bad = []
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.stmt | ast.expr):
                continue
            if (
                isinstance(node, ast.Call)
                and (name := _dotted(node.func)) in FORBIDDEN_CALLS
                and not (name in PROCESS_FLAG_CALLS and path.relative_to(ROOT).as_posix() == PROCESS_FLAGS)
            ):
                bad.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name} {FORBIDDEN_CALLS[name]}")
            if (
                isinstance(node, ast.Call)
                and (name := _dotted(node.func) or "").startswith("torch.backends.")
                and path.relative_to(ROOT).as_posix() != PROCESS_FLAGS
                and any(part.startswith(("enable_", "allow_")) for part in name.split("."))
            ):
                bad.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name} changes a process-wide flag")
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
            for target in targets:
                name = _dotted(target.value if isinstance(target, ast.Subscript) else target) or ""
                if name == "os.environ" or (name.startswith(("torch.backends.", "torch._inductor.config")) and path.relative_to(ROOT).as_posix() != PROCESS_FLAGS):
                    bad.append(f"{path.relative_to(ROOT)}:{node.lineno}: assigns {name}")
    return bad


def test_native_code_leaves_global_state_alone() -> None:
    """Engines share the process, so they change nothing process-wide."""
    bad = _global_state_changes([p for layer in ("runtime", "nn", "text", "engines") for p in (HOST / layer).rglob("*.py")])
    assert not bad, "\n".join(bad)


def _attention_outside_accel(paths: list[Path]) -> list[str]:
    """SDPA or its kernels called in ``paths`` outside ``mizuoto.runtime.accel``."""
    bad = []
    for path in paths:
        if (HOST / "runtime" / "accel") in path.parents:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and (node.attr == "scaled_dot_product_attention" or node.attr.startswith("_scaled_dot_product")):
                bad.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.attr}")
    return bad


def test_attention_goes_through_the_accel_entry_point() -> None:
    """Every attention computation is a site of ``mizuoto.runtime.accel``, so the backend
    switch reaches it; SDPA and its kernels are called only there."""
    bad = _attention_outside_accel([p for layer in ("runtime", "nn", "engines", "training") for p in (HOST / layer).rglob("*.py")])
    assert not bad, "\n".join(bad)


def test_host_never_imports_model_code() -> None:
    """The host's shells and services; the native runtime and engines are where model code lives."""
    host = [p for layer in ("core", "api", "cli") for p in (HOST / layer).rglob("*.py")]
    bad = [f"{p.relative_to(ROOT)}: {name}" for p in host for name, _ in _imports(p) if name.split(".")[0] in NEVER_IN_HOST | {"mizuoto_fishspeech", "mizuoto_indextts"}]
    assert not bad, "\n".join(bad)


def _loaded(code: str) -> str:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()


def test_core_import_is_light() -> None:
    assert _loaded("import sys, mizuoto.core.context; print('torch' in sys.modules, 'fastapi' in sys.modules, 'typer' in sys.modules)") == "False False False"


def test_engine_contract_import_is_light() -> None:
    """The contract, the request context and the dummy engine load no torch: the host's tests need none."""
    code = "import sys, mizuoto.runtime.engine, mizuoto.runtime.context, mizuoto.engines.dummy, mizuoto.core.inference.local, mizuoto.core.inference.runtime; print('torch' in sys.modules)"
    assert _loaded(code) == "False"


def test_cli_import_is_light() -> None:
    assert _loaded("import sys, mizuoto.cli.app; print('torch' in sys.modules, 'fastapi' in sys.modules)") == "False False"


def test_one_distribution() -> None:
    """Mizuoto is one distribution: no worker package, and every engine's dependencies are its own."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert not [d for d in project["dependencies"] if d.startswith("mizuoto-") or d.split(">")[0].split("=")[0] == "uv"]
    assert any(d.startswith("torch") for d in project["dependencies"]) and "native" not in project.get("optional-dependencies", {})
    assert not (ROOT / "packages" / "mizuoto-worker").exists() and not (ROOT / "packages" / "engines").exists()
