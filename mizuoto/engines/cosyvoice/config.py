"""A CosyVoice model folder's hyperparameters, read from its HyperPyYAML file without HyperPyYAML.

``cosyvoice.yaml``, ``cosyvoice2.yaml`` and ``cosyvoice3.yaml`` build the model with tags (``!new:``,
``!name:``, ``!ref <key>``, ``!apply:``). Here the tags are kept as data: a ``!new:``/``!name:`` mapping
becomes a dict with its class under ``"_class"``, and ``!ref`` expressions (``<chunk_size> * <token_mel_ratio>``)
are resolved against the top-level keys. Only the numbers the native modules need are read from it.
"""

from __future__ import annotations

import ast
import operator
import re
from pathlib import Path
from typing import Any

YAML_NAMES = ("cosyvoice3.yaml", "cosyvoice2.yaml", "cosyvoice.yaml")


class _Ref(str):
    pass


def _loader() -> Any:
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def tagged(loader: Any, suffix: str, node: Any) -> Any:
        if isinstance(node, yaml.MappingNode):
            value = loader.construct_mapping(node, deep=True)
            return {"_class": suffix, **value}
        if isinstance(node, yaml.SequenceNode):
            return {"_class": suffix, "_args": loader.construct_sequence(node, deep=True)}
        return {"_class": suffix, "_value": loader.construct_scalar(node)}

    def ref(loader: Any, node: Any) -> Any:
        return _Ref(loader.construct_scalar(node))

    Loader.add_multi_constructor("!new:", tagged)
    Loader.add_multi_constructor("!name:", tagged)
    Loader.add_multi_constructor("!apply:", tagged)
    Loader.add_constructor("!ref", ref)
    return Loader


_OPS: dict[type, Any] = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv}


def _eval(expr: str) -> Any:
    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError(f"Unsupported expression {expr!r}")

    return ev(ast.parse(expr, mode="eval"))


def _resolve(value: Any, top: dict[str, Any]) -> Any:
    if isinstance(value, _Ref):
        names = re.findall(r"<([^>]+)>", value)
        if len(names) == 1 and value.strip() == f"<{names[0]}>":
            return _resolve(top.get(names[0]), top)
        text = value
        for name in names:
            text = text.replace(f"<{name}>", repr(_resolve(top.get(name), top)))
        return _eval(text)
    if isinstance(value, dict):
        return {k: _resolve(v, top) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, top) for v in value]
    return value


def read_config(model_dir: Path) -> dict[str, Any]:
    """The model's yaml as plain data (``llm``, ``flow``, ``hift``, ``feat_extractor``, ``sample_rate``…)."""
    import yaml

    path = next((model_dir / n for n in YAML_NAMES if (model_dir / n).is_file()), None)
    if path is None:
        raise FileNotFoundError(f"{model_dir} has no CosyVoice yaml")
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_loader())  # a SafeLoader subclass
    return {k: _resolve(v, raw) for k, v in raw.items() if not k.startswith("__")}


def kwargs(node: dict[str, Any], *names: str) -> dict[str, Any]:
    """The constructor arguments ``names`` present in a ``!new:`` node."""
    return {n: node[n] for n in names if n in node}
