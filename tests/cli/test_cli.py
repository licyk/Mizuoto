import importlib.util
import json
import logging
from typing import Any

import pytest
from typer.testing import CliRunner

from mizuoto.cli.app import get_app
from mizuoto.logger import LOGGER_NAME

EXPECTED_TREE = {
    "agent": None,
    "asr": None,
    "asset": {"delete": None, "download": None, "list": None},
    "config": {"get": None, "path": None, "set": None, "show": None},
    "convert": None,
    "dataset": {k: None for k in ("add", "check", "create", "delete", "edit-row", "import", "info", "list", "prepare", "rows")},
    "design": None,
    "doctor": None,
    "edit": None,
    "engine": {"accept-licence": None, "disable": None, "enable": None, "info": None, "licence": None, "list": None, "op": None},
    "env": {"vars": None},
    "job": {"cancel": None, "list": None, "log": None, "retry": None, "show": None},
    "model": {"delete": None, "download": None, "import": None, "info": None, "list": None, "spec": None, "speakers": None, "verify": None},
    "output": {"check": None, "delete": None, "info": None, "list": None},
    "preset": {"default": None, "delete": None, "list": None, "save": None},
    "project": {k: None for k in ("cast", "delete", "export", "import", "list", "new", "render", "script", "set", "show")},
    "remote": {"add": None, "list": None, "remove": None, "use": None},
    "runtime": None,
    "speak": None,
    "train": {k: None for k in ("delete", "edit", "export", "list", "methods", "metrics", "new", "run", "status")},
    "version": None,
    "webui": None,
    "voice": {
        "add": None,
        "add-ref": None,
        "add-weights": None,
        "design": None,
        "edit": None,
        "export": None,
        "fit": None,
        "from-output": None,
        "from-preset": None,
        "import": None,
        "info": None,
        "list": None,
        "prepare": None,
        "remove": None,
        "remove-ref": None,
        "remove-weights": None,
        "retranscribe": None,
        "set-transcript": None,
    },
}


def _tree(group: Any) -> dict[str, Any]:
    import typer

    click_group: Any = typer.main.get_command(group) if not hasattr(group, "commands") else group
    return {name: (_tree(cmd) if hasattr(cmd, "commands") else None) for name, cmd in click_group.commands.items()}


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("MIZUOTO_DATA_DIR", str(tmp_path / "data"))
    logger = logging.getLogger(LOGGER_NAME)
    level = logger.level
    yield CliRunner()
    logger.setLevel(level)


def test_command_tree():
    assert _tree(get_app()) == EXPECTED_TREE


def test_debug_everywhere(runner):
    app = get_app()
    for args in (["--debug", "version"], ["config", "--debug", "path"], ["config", "show", "--debug"]):
        r = runner.invoke(app, args)
        assert r.exit_code == 0, (args, r.output)


def test_version_and_config(runner, tmp_path):
    app = get_app()
    r = runner.invoke(app, ["version", "--json"])
    assert r.exit_code == 0 and json.loads(r.stdout)["mizuoto"]
    assert runner.invoke(app, ["config", "set", "server.port", "9001"]).exit_code == 0
    assert json.loads(runner.invoke(app, ["config", "get", "server.port"]).stdout) == 9001
    assert runner.invoke(app, ["config", "path"]).stdout.strip() == str((tmp_path / "data" / "settings.toml").resolve())
    shown = json.loads(runner.invoke(app, ["config", "show", "--json"]).stdout)
    assert shown["server"]["port"] == 9001 and "access_token" not in shown["server"]
    r = runner.invoke(app, ["config", "set", "nope.key", "1"])
    assert r.exit_code == 4


def test_speak_writes_the_file(runner, tmp_path):
    app = get_app()
    target = tmp_path / "out" / "hello.mp3"
    r = runner.invoke(app, ["speak", "Hello there.", "-m", "dummy/default", "--preset", "alto", "--seed", "2", "-o", str(target), "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert data["job"]["state"] == "completed" and data["outputs"][0]["format"] == "mp3"
    assert target.is_file() and target.stat().st_size > 1000


def test_speak_plan_and_errors(runner):
    app = get_app()
    r = runner.invoke(app, ["speak", "Hi. There.", "-m", "dummy/default", "--plan", "--json"])
    assert r.exit_code == 0 and json.loads(r.stdout)["voice_mode"] == "auto"
    # A parameter the model does not take is dropped with a warning unless --strict asks to refuse it.
    r = runner.invoke(app, ["speak", "x", "-m", "dummy/default", "-p", "top_k=3", "--plan", "--json"])
    assert r.exit_code == 0 and "top_k" not in json.loads(r.stdout)["params"]
    assert runner.invoke(app, ["speak", "x", "-m", "dummy/default", "-p", "top_k=3", "--strict"]).exit_code == 4
    assert runner.invoke(app, ["speak", "x", "-m", "dummy/nope"]).exit_code == 2
    assert runner.invoke(app, ["speak", "x", "-m", "dummy/default", "-O", "fail=error"]).exit_code == 10


def test_listings_print_json(runner):
    app = get_app()
    engines = json.loads(runner.invoke(app, ["engine", "list", "--json"]).stdout)
    assert {"asr", "dummy", "qwen3tts", "omnivoice", "cosyvoice", "f5tts"} <= {e["id"] for e in engines}
    by_id = {e["id"]: e for e in engines}
    # Every engine runs in this process; it is available where torch is installed.
    torch = importlib.util.find_spec("torch") is not None
    assert by_id["dummy"]["available"] and by_id["asr"]["available"] == torch and by_id["asr"]["internal"]
    assert by_id["asr"]["missing"] == ([] if torch else ["torch"])
    models = {m["id"]: m for m in json.loads(runner.invoke(app, ["model", "list", "--json"]).stdout)}
    assert models["dummy/default"]["state"] == "builtin" and models["qwen3tts/1.7b-base"]["state"] == "missing"
    assert models["qwen3tts/1.7b-base"]["size_mb"] > 4000
    spec = json.loads(runner.invoke(app, ["model", "spec", "dummy/default", "--json"]).stdout)
    assert spec["sample_rate"] == 24000


def test_voice_commands(runner, wav_file, tmp_path):
    app = get_app()
    r = runner.invoke(app, ["voice", "add", "Test Voice", "--ref", str(wav_file), "--text", "Hello.", "--lang", "en", "--consent", "mine", "--json"])
    assert r.exit_code == 0, r.output
    voice = json.loads(r.stdout)
    assert voice["slug"] == "test-voice" and voice["references"][0]["transcript"] == "Hello."
    listed = json.loads(runner.invoke(app, ["voice", "list", "--json"]).stdout)
    assert [v["slug"] for v in listed] == ["test-voice"]
    fit = json.loads(runner.invoke(app, ["voice", "fit", "test-voice", "-m", "dummy/default", "--json"]).stdout)
    assert fit["usable"] and fit["mode"] == "clone"
    target = tmp_path / "out.wav"
    r = runner.invoke(app, ["speak", "Hi.", "-m", "dummy/default", "--voice", "test-voice", "-o", str(target)])
    assert r.exit_code == 0, r.output
    assert target.is_file()
    assert runner.invoke(app, ["voice", "from-preset", "dummy/default", "alto"]).exit_code == 0
    r = runner.invoke(app, ["design", "Designed.", "-m", "dummy/default", "-d", "a bright voice", "--save-as", "Bright", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["voice"]["kind"] == "designed"
    archive = tmp_path / "v.zip"
    assert runner.invoke(app, ["voice", "export", "test-voice", "-o", str(archive)]).exit_code == 0
    assert runner.invoke(app, ["voice", "import", str(archive), "--name", "Copy"]).exit_code == 0
    assert runner.invoke(app, ["voice", "remove", "copy"]).exit_code == 0
    assert runner.invoke(app, ["voice", "info", "nobody"]).exit_code == 2


def test_doctor_runtime_and_model_commands(runner, wav_file, tmp_path):
    app = get_app()
    r = runner.invoke(app, ["doctor", "qwen3tts", "--json"])
    report = json.loads(r.stdout)
    assert report["recommended_build"] and "pip install" in report["install_command"]
    engine = [f for f in report["findings"] if f["engine"] == "qwen3tts"]
    assert [f["check"] for f in engine] == ["engine"] and {f["engine"] for f in report["findings"]} <= {None, "qwen3tts"}
    assert r.exit_code == (1 if any(f["level"] == "error" for f in report["findings"]) else 0)
    assert runner.invoke(app, ["doctor", "nope"]).exit_code == 2
    runtime = json.loads(runner.invoke(app, ["runtime", "--json"]).stdout)
    assert runtime["models"] == [] and runtime["engines"] == []
    assert runner.invoke(app, ["runtime", "--unload", "dummy/default"]).exit_code == 2
    assert runner.invoke(app, ["runtime", "dummy/default"]).exit_code == 2
    assert runner.invoke(app, ["model", "download", "dummy/nope"]).exit_code == 2
    assert runner.invoke(app, ["speak", "x", "-m", "qwen3tts/1.7b-base"]).exit_code == 7
    assert runner.invoke(app, ["asr", __file__]).exit_code == 4
    assert runner.invoke(app, ["asr", str(wav_file)]).exit_code == 7
    assert "PYTHONPATH" not in runner.invoke(app, ["env", "vars"]).stdout


def test_presets_and_streaming(runner, tmp_path):
    app = get_app()
    r = runner.invoke(app, ["preset", "save", "Slow", "-m", "dummy/default", "-p", "speed=0.8", "--default", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["is_default"] is True
    listed = json.loads(runner.invoke(app, ["preset", "list", "--json"]).stdout)
    assert [p["name"] for p in listed] == ["Slow"]
    r = runner.invoke(app, ["speak", "Hi.", "-m", "dummy/default", "-P", "Slow", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["outputs"][0]["params"]["speed"] == 0.8
    assert runner.invoke(app, ["preset", "default", "Slow", "--off"]).exit_code == 0
    assert runner.invoke(app, ["preset", "delete", "Slow"]).exit_code == 0
    assert runner.invoke(app, ["preset", "delete", "Slow"]).exit_code == 2
    r = runner.invoke(app, ["speak", "Hello there.", "-m", "dummy/default", "--stream"])
    assert r.exit_code == 0, r.output
    assert r.stdout_bytes[:4] == b"RIFF" and len(r.stdout_bytes) > 24000


def test_engine_op(runner):
    r = runner.invoke(get_app(), ["engine", "op", "dummy/default", "echo", "-s", "x=2", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["result"]["echo"] == {"x": 2}


def test_dataset_and_train(runner, tmp_path):
    from tests.core.test_training import tone

    app = get_app()
    folder = tmp_path / "clips"
    folder.mkdir()
    for i in range(3):
        tone(folder / f"c{i}.wav", 3.0 + i)
        (folder / f"c{i}.lab").write_text(f"Clip {i}.")
    assert runner.invoke(app, ["dataset", "create", "Tones", "--lang", "en"]).exit_code == 0
    r = runner.invoke(app, ["dataset", "add", "tones", str(folder)])
    assert r.exit_code == 0 and "3 clip(s)" in r.stdout, r.output
    r = runner.invoke(app, ["dataset", "prepare", "tones", "--no-transcribe"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["dataset", "check", "tones"])
    assert r.exit_code == 0 and "3 of 3 clip(s) usable" in r.stdout
    rows = json.loads(runner.invoke(app, ["dataset", "rows", "tones", "--json"]).stdout)
    assert runner.invoke(app, ["dataset", "edit-row", "tones", rows["items"][0]["id"], "--exclude"]).exit_code == 0
    assert "speaker" in runner.invoke(app, ["train", "methods", "dummy/default"]).stdout
    r = runner.invoke(app, ["train", "new", "t1", "--base", "dummy/default", "--dataset", "tones", "--hp", "epochs=1"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["train", "run", "t1"])
    assert r.exit_code == 0 and "Exported voice" in r.stdout, r.output
    assert "loss" in runner.invoke(app, ["train", "metrics", "t1"]).stdout
    r = runner.invoke(app, ["train", "run", "t1"])
    assert r.exit_code == 4 and "--restart" in r.output
    assert runner.invoke(app, ["train", "edit", "t1", "--hp", "epochs=2"]).exit_code == 0
    exp = json.loads(runner.invoke(app, ["train", "status", "t1", "--json"]).stdout)
    assert [s["status"] for s in exp["stages"]] == ["done", "stale", "stale"]
    assert runner.invoke(app, ["train", "delete", "dummy/t1", "-y"]).exit_code == 0
    assert runner.invoke(app, ["dataset", "delete", "tones", "-y"]).exit_code == 0


def test_data_dir_option_before_or_after_the_command(runner, tmp_path):
    """``--data-dir`` at any level points the command at another data directory, as the environment variable does."""
    app = get_app()
    for args, folder in ((["--data-dir", str(tmp_path / "a"), "config", "path"], "a"), (["config", "path", "--data-dir", str(tmp_path / "b")], "b")):
        r = runner.invoke(app, args)
        assert r.exit_code == 0, r.output
        assert r.output.strip().endswith(str(tmp_path / folder / "settings.toml")), r.output
    r = runner.invoke(app, [*["output", "list", "--json"], "--data-dir", str(tmp_path / "c")])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "c" / "mizuoto.db").is_file() and not (tmp_path / "data" / "mizuoto.db").exists()
