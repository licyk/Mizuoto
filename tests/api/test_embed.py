"""Embedding Mizuoto in another application: MizuotoServer, the configuration folder, the route prefix, mounting."""

import json
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mizuoto import MizuotoServer
from mizuoto.api.app import create_app, normalize_prefix
from mizuoto.core.context import build_services
from mizuoto.core.events.models import SettingsChangedEvent
from mizuoto.core.net.runtime_file import read_runtime_file
from mizuoto.core.settings import SettingsService


def get(url: str, timeout: float = 10.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_start_serves_and_stops(tmp_path):
    server = MizuotoServer(data_dir=tmp_path / "data", port=0)
    url = server.start()
    try:
        assert url.startswith("http://127.0.0.1:") and server.running
        status, body = get(f"{url}/api/v1/app/health")
        assert status == 200 and '"status":"ok"' in body.replace(" ", "")
        # A command line opened on the same data directory sees that a server owns its jobs.
        runtime = read_runtime_file(tmp_path / "data")
        assert runtime is not None and runtime["url"] == url
    finally:
        server.stop()
    assert not server.running and read_runtime_file(tmp_path / "data") is None


def test_stop_releases_the_port_and_is_repeatable(tmp_path):
    server = MizuotoServer(data_dir=tmp_path / "data", port=0)
    server.start()
    port = server.port
    server.stop()
    server.stop()
    with pytest.raises(RuntimeError):
        _ = server.url
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_a_taken_port_moves_up_unless_strict(tmp_path):
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        moved = MizuotoServer(data_dir=tmp_path / "a", port=port)
        try:
            moved.start()
            assert moved.port != port
        finally:
            moved.stop()
        strict = MizuotoServer(data_dir=tmp_path / "b", port=port, strict_port=True)
        with pytest.raises(Exception, match="strict|unavailable|Cannot bind"):
            strict.start()
        strict.stop()


def test_a_remote_host_needs_no_token(tmp_path):
    server = MizuotoServer(data_dir=tmp_path / "data", host="0.0.0.0", port=0)
    server.start()
    try:
        status, body = get(f"http://127.0.0.1:{server.port}/api/v1/app/health")
        assert status == 200 and '"auth_required":false' in body.replace(" ", "")
    finally:
        server.stop()


def test_allow_host_while_serving(tmp_path):
    # A tunnel connects from loopback and forwards its public Host, as Gradio's share links do.
    server = MizuotoServer(data_dir=tmp_path / "data", port=0)
    url = server.start()
    try:

        def request(host: str, path: str = "/api/v1/app/meta") -> tuple[int, str]:
            try:
                with urllib.request.urlopen(urllib.request.Request(f"{url}{path}", headers={"Host": host}), timeout=10) as response:
                    return response.status, response.read().decode()
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode()

        status, body = request("abc.gradio.live")
        assert status == 400 and "bad_host" in body
        server.allow_host("https://ABC.gradio.live/voice")
        status, body = request("abc.gradio.live")
        # Its visitors are elsewhere: they get no local-only operations, but need no token.
        assert status == 200 and '"local":false' in body.replace(" ", "")
        # Server paths are for a browser on the server's own machine only.
        status, body = request("abc.gradio.live", "/api/v1/audio/browse")
        assert status == 403 and "not_local" in body
        status, body = request("localhost")
        assert status == 200 and '"local":true' in body.replace(" ", "")
    finally:
        server.stop()


def test_extra_hosts_take_an_optional_token(tmp_path):
    with MizuotoServer(data_dir=tmp_path / "data", port=0, extra_hosts=["abc.example.com"], access_token="secret") as server:

        def status(headers: dict[str, str]) -> int:
            try:
                with urllib.request.urlopen(urllib.request.Request(f"{server.url}/api/v1/app/meta", headers={"Host": "abc.example.com", **headers}), timeout=10) as response:
                    return response.status
            except urllib.error.HTTPError as e:
                return e.code

        assert status({}) == 401
        assert status({"Authorization": "Bearer secret"}) == 200


def test_config_dir_holds_the_settings_file(tmp_path):
    config = tmp_path / "host" / "config"
    with MizuotoServer(data_dir=tmp_path / "data", config_dir=config, port=0) as server:
        services = server.services
        assert services is not None
        services.settings.update({"outputs": {"keep_days": 3}})
        assert services.settings.path == (config / "settings.toml").resolve()
        assert services.settings.view().settings_file == str(services.settings.path)
        _, body = get(f"{server.url}/api/v1/settings")
        assert json.loads(body)["settings_file"] == str(config / "settings.toml")  # parsed: JSON escapes Windows' backslashes
    assert (config / "settings.toml").is_file()
    assert not (tmp_path / "data" / "settings.toml").exists()
    assert (tmp_path / "data" / "mizuoto.db").is_file()
    # The next start reads it from there.
    again = SettingsService(data_dir=tmp_path / "data", config_dir=config, environ={})
    assert again.settings.outputs.keep_days == 3


def test_settings_path_names_the_file(tmp_path):
    path = tmp_path / "host" / "voice.toml"
    services = MizuotoServer(data_dir=tmp_path / "data", config_dir=tmp_path / "ignored", settings_path=path, port=0)._build()
    try:
        services.settings.update({"outputs": {"keep_days": 5}})
        assert path.is_file() and not (tmp_path / "ignored").exists()
    finally:
        services.close()


def test_config_dir_from_the_environment(tmp_path):
    settings = SettingsService(data_dir=tmp_path / "data", environ={"MIZUOTO_CONFIG_DIR": str(tmp_path / "cfg")})
    assert settings.path == (tmp_path / "cfg" / "settings.toml").resolve()
    assert settings.env_override_names() == []
    assert SettingsService(data_dir=tmp_path / "data", environ={}).path == (tmp_path / "data" / "settings.toml").resolve()


def test_pinned_settings_stay(tmp_path):
    pins = {"outputs": {"keep_days": 1}, "paths": {"models_dir": str(tmp_path / "voices")}}
    services = MizuotoServer(data_dir=tmp_path / "data", port=0, settings=pins)._build()
    try:
        services.settings.update({"outputs": {"keep_days": 9}, "tts": {"output_format": "flac"}})
        assert services.settings.settings.outputs.keep_days == 1
        assert services.settings.settings.tts.output_format == "flac"
        assert services.settings.models_dir == (tmp_path / "voices").resolve()
        pinned = services.settings.view().pinned
        assert "outputs.keep_days" in pinned and "paths.models_dir" in pinned and "tts.output_format" not in pinned
    finally:
        services.close()


@pytest.mark.parametrize(("given", "expected"), [(None, ""), ("", ""), ("/voice", "/voice"), ("voice", "/voice"), ("/voice/", "/voice"), ("a/b", "/a/b")])
def test_prefix_is_normalized(given, expected):
    assert normalize_prefix(given) == expected


def test_server_url_carries_the_prefix(tmp_path):
    with MizuotoServer(data_dir=tmp_path / "data", port=0, api_prefix="voice") as server:
        assert server.url.endswith("/voice")
        assert get(f"{server.url}/api/v1/app/health")[0] == 200
        assert get(server.url.removesuffix("/voice") + "/api/v1/app/health")[0] == 404


def test_everything_moves_under_the_prefix(tmp_path):
    services = build_services(data_dir=tmp_path / "data", environ={})
    app = create_app(services, bound_host="127.0.0.1", serve_ui=False, api_prefix="/voice")
    try:
        with TestClient(app, base_url="http://localhost") as client:
            assert client.get("/voice/api/v1/app/health").status_code == 200
            assert client.get("/voice/openapi.json").status_code == 200
            reply = client.get("/voice/ws/socket.io/", params={"EIO": "4", "transport": "polling"})
            assert reply.status_code == 200 and reply.text.startswith("0{")
            assert client.get("/api/v1/app/health").status_code == 404
    finally:
        services.close()


@pytest.mark.parametrize(("mount", "prefix"), [("", "/voice"), ("/host", ""), ("/host", "/voice")])
def test_mounted_api_security_and_socket(tmp_path, mount, prefix):
    services = build_services(data_dir=tmp_path / "data", environ={})
    services.settings.update({"server": {"access_token": "secret"}})
    child = create_app(services, bound_host="localhost", api_prefix=prefix, serve_ui=False)

    @asynccontextmanager
    async def lifespan(_app):
        async with child.router.lifespan_context(child):
            yield

    parent = FastAPI(lifespan=lifespan)
    parent.mount(mount or "/", child)
    base = mount + prefix
    try:
        with TestClient(parent, base_url="http://localhost") as client:
            assert client.get(f"{base}/api/v1/app/health").status_code == 200
            assert client.get(f"{base}/api/v1/settings").status_code == 401
            headers = {"Authorization": "Bearer secret"}
            assert client.get(f"{base}/api/v1/settings", headers=headers).status_code == 200
            assert client.patch(f"{base}/api/v1/settings", json={}, headers={**headers, "Origin": "https://elsewhere.example"}).status_code == 403
            with client.websocket_connect(f"ws://localhost{base}/ws/socket.io/?EIO=4&transport=websocket", headers={**headers, "Upgrade": "websocket"}) as ws:
                assert ws.receive_text().startswith('0{"sid":')
                ws.send_text("40")
                assert ws.receive_text().startswith("40")
                services.events.publish(SettingsChangedEvent(keys=["tts.output_format"]))
                message = ws.receive_text()
                assert message.startswith('42["settings_changed",') and "tts.output_format" in message
    finally:
        services.close()


def test_prefixed_ui_is_served(tmp_path, monkeypatch):
    from mizuoto.api import app as app_module

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<script src="./assets/app.js"></script>')
    (dist / "assets" / "app.js").write_text("window.loaded = true;")
    monkeypatch.setattr(app_module, "web_dist_dir", lambda: dist)
    services = build_services(data_dir=tmp_path / "data", environ={})
    try:
        app = create_app(services, bound_host="127.0.0.1", api_prefix="/voice")
        with TestClient(app, base_url="http://localhost") as client:
            assert client.get("/voice/", headers={"accept": "text/html"}).text.startswith("<script")
            assert client.get("/voice/assets/app.js").text == "window.loaded = true;"
            assert client.get("/voice/api/v1/app/health").status_code == 200
    finally:
        services.close()


def test_import_is_light():
    """``import mizuoto`` and the server class stay free of torch, as the command line does."""
    code = "import sys, mizuoto; mizuoto.MizuotoServer; assert 'torch' not in sys.modules, 'torch'"
    subprocess.run([sys.executable, "-c", code], check=True)
