"""scripts/ci.py: which checks a change runs, and which distributions a run of Release publishes."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("ci_script", ROOT / "scripts" / "ci.py")
assert _spec and _spec.loader
ci = importlib.util.module_from_spec(_spec)
sys.modules["ci_script"] = ci  # its dataclasses look their module up
_spec.loader.exec_module(ci)

GATED = ["fishspeech", "indextts"]


def test_each_distribution_is_checked_on_its_own():
    only = ci.scope(["packages/native/fishspeech/mizuoto_fishspeech/engine.py", "packages/native/fishspeech/tests/test_fishspeech_tiny.py"])
    assert only == {"core": False, "web": False, "site": False, "gated": ["fishspeech"], "release_files": False}
    assert ci.scope(["packages/native/indextts/pyproject.toml"])["gated"] == ["indextts"]
    assert ci.scope(["site/content/docs/index.mdx"]) == {"core": False, "web": False, "site": True, "gated": [], "release_files": False}
    assert ci.scope(["mizuoto/webui/src/App.vue"]) == {"core": False, "web": True, "site": False, "gated": [], "release_files": False}
    assert ci.scope(["README.md", "AGENTS.md", "Dockerfile"]) == {"core": False, "web": False, "site": False, "gated": [], "release_files": False}


def test_a_change_to_mizuoto_checks_the_packages_that_run_on_it():
    for path in ("mizuoto/runtime/engine.py", "pyproject.toml", "tests/engines_native/streaming.py", "scripts/build_wheels.py", "something/new.txt"):
        assert ci.scope([path]) == {"core": True, "web": True, "site": False, "gated": GATED, "release_files": False}, path


def test_a_change_to_ci_checks_everything():
    for files, everything in (([".github/workflows/ci.yml"], False), (["scripts/ci.py"], False), ([], True)):
        assert ci.scope(files, everything) == {"core": True, "web": True, "site": True, "gated": GATED, "release_files": False}


def test_version_files_mark_a_release():
    assert ci.scope(["mizuoto/version.py"])["release_files"]
    assert ci.scope(["packages/native/indextts/mizuoto_indextts/version.py"])["release_files"]
    assert not ci.scope(["packages/native/indextts/mizuoto_indextts/__init__.py"])["release_files"]


def test_release_publishes_what_the_push_tag_or_input_names():
    version = {d.name: ci.version_of(d) for d in ci.distributions()}
    pushed = ci.release("push", "branch", "main", ["packages/native/indextts/mizuoto_indextts/version.py", "mizuoto/core/x.py"])
    assert [d.name for d in pushed] == ["mizuoto-indextts"]
    assert [d.name for d in ci.release("push", "branch", "main", ["mizuoto/version.py"])] == ["mizuoto"]
    assert [d.name for d in ci.release("push", "tag", f"v{version['mizuoto']}", [])] == ["mizuoto"]
    tagged = ci.release("push", "tag", f"mizuoto-fishspeech-v{version['mizuoto-fishspeech']}", [])
    assert [d.name for d in tagged] == ["mizuoto-fishspeech"]
    assert [d.name for d in ci.release("workflow_dispatch", "branch", "main", [], "all")] == ["mizuoto", "mizuoto-fishspeech", "mizuoto-indextts"]
    assert [d.name for d in ci.release("workflow_dispatch", "branch", "main", [], "mizuoto-indextts, mizuoto")] == ["mizuoto", "mizuoto-indextts"]
    with pytest.raises(SystemExit, match="does not match"):
        ci.release("push", "tag", "mizuoto-indextts-v999.0.0", [])
    with pytest.raises(SystemExit, match="names no distribution"):
        ci.release("push", "tag", "release-1", [])
    with pytest.raises(SystemExit, match="Unknown"):
        ci.release("workflow_dispatch", "branch", "main", [], "mizuoto-nope")


def test_lower_bounds_are_released_versions():
    for engine in GATED:
        assert re.fullmatch(r"\d+\.\d+\.\d+", ci.lower_bound(engine)), engine


def test_workflows_name_the_version_files():
    """Release's path filter lists every version file, so a push that changes one releases it."""
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert '"mizuoto/version.py"' in release and '"packages/native/*/mizuoto_*/version.py"' in release
    assert all(Path(ROOT / d.version_file).is_file() for d in ci.distributions())
