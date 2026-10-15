import os
import time

import pytest

from mizuoto.core.errors import ValidationError
from mizuoto.core.settings import SettingsService
from mizuoto.core.settings.models import DEFAULT_PORT


def test_defaults_and_data_dir(tmp_path):
    s = SettingsService(data_dir=tmp_path, environ={})
    assert s.settings.server.port == DEFAULT_PORT == 7869
    assert s.path == tmp_path / "settings.toml"
    assert s.cache_dir == tmp_path / "cache"
    assert s.voices_dir == tmp_path / "voices"
    view = s.view()
    assert view.server.access_token_configured is False
    assert view.resolved_paths.models_dir == str(tmp_path / "models")


def test_precedence_file_env_overrides(tmp_path):
    (tmp_path / "settings.toml").write_text("[server]\nport = 8001\nhost = '0.0.0.0'\n")
    env = {"MIZUOTO_SERVER__PORT": "8002", "MIZUOTO_DOWNLOADS__SOURCE": "hf-mirror"}
    s = SettingsService(data_dir=tmp_path, environ=env, overrides={"server": {"host": "127.0.0.2"}})
    assert s.settings.server.port == 8002
    assert s.settings.server.host == "127.0.0.2"
    assert s.settings.downloads.source == "hf-mirror"
    assert s.view().env_overrides == ["MIZUOTO_DOWNLOADS__SOURCE", "MIZUOTO_SERVER__PORT"]
    assert s.view().pinned == ["server.host"]


def test_update_writes_file_and_notifies(tmp_path):
    s = SettingsService(data_dir=tmp_path, environ={})
    seen: list[list[str]] = []
    s.on_keys_change(seen.append)
    s.update({"tts": {"output_format": "flac"}, "engines": {"qwen3tts": {"device": "cuda:1"}}})
    assert s.settings.tts.output_format == "flac"
    assert s.settings.engines["qwen3tts"].device == "cuda:1"
    text = (tmp_path / "settings.toml").read_text()
    assert 'output_format = "flac"' in text
    assert seen and "tts.output_format" in seen[0]


def test_set_value_and_unknown_key(tmp_path):
    s = SettingsService(data_dir=tmp_path, environ={})
    s.set_value("server.port", "9001")
    assert s.get_value("server.port") == 9001
    s.set_value("engines.cosyvoice", '{"enabled": false}')
    assert s.settings.engines["cosyvoice"].enabled is False
    with pytest.raises(ValidationError):
        s.set_value("nope.key", "1")
    with pytest.raises(ValidationError):
        s.set_value("server.port", "0")


def test_change_by_another_process_is_merged(tmp_path):
    a = SettingsService(data_dir=tmp_path, environ={})
    a.update({"server": {"port": 8100}})
    b = SettingsService(data_dir=tmp_path, environ={})
    b.update({"tts": {"normalize": "loudness"}})
    # a has not seen b's change yet; its next save must keep it
    stamp = (tmp_path / "settings.toml").stat().st_mtime_ns
    os.utime(tmp_path / "settings.toml", ns=(stamp + 10**9, stamp + 10**9))
    time.sleep(0.01)
    a.update({"server": {"strict_port": True}})
    c = SettingsService(data_dir=tmp_path, environ={})
    assert c.settings.tts.normalize == "loudness"
    assert c.settings.server.port == 8100 and c.settings.server.strict_port


def test_secrets_are_hidden(tmp_path):
    s = SettingsService(data_dir=tmp_path, environ={})
    s.update({"server": {"access_token": "tok-9f3a7c"}})
    dumped = s.view().model_dump_json()
    assert "tok-9f3a7c" not in dumped
    assert s.view().server.access_token_configured is True


def test_data_dir_from_injected_environ(tmp_path):
    s = SettingsService(environ={"MIZUOTO_DATA_DIR": str(tmp_path / "d")})
    assert s.data_dir == (tmp_path / "d").resolve()
