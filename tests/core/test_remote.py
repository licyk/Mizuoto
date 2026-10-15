"""A host driving an agent's engines (both on this machine, over TCP), with the dummy engine."""

import threading
import time

import pytest

from mizuoto.core.errors import DeviceError, RemoteError
from mizuoto.core.remote.agent import AgentServer
from mizuoto.core.tts.models import CommonParams, ReferenceInput, SpeakRequest, VoiceInput
from mizuoto.core.tts.service import StreamSink

MODEL = "dummy/default"


@pytest.fixture
def agent(tmp_path):
    from mizuoto.core.context import build_services

    services = build_services(data_dir=tmp_path / "agent", environ={}, start_background=True)
    server = AgentServer(services, "127.0.0.1", 0, "s3cret", heartbeat=0.5)
    server.start()
    yield server
    server.close()
    services.close()


def connect(make_services, agent, token="s3cret", start_background=True):
    services = make_services(start_background=start_background)
    port = agent.address[1]
    services.settings.update({"remote": {"hosts": {"box": {"address": f"127.0.0.1:{port}", "token": token}}, "heartbeat_timeout": 10}, "engines": {"dummy": {"device": "box/cpu"}}})
    return services


def test_status_and_refusal(make_services, agent):
    services = connect(make_services, agent)
    [status] = services.remote.status()
    assert status.state == "online" and status.engines == ["dummy"]
    assert status.info is not None and MODEL in status.info.models and any(e.id == "dummy" and e.ready for e in status.info.engines)
    assert services.engines.status("dummy").remote == "box"
    assert services.models.get(MODEL).state == "builtin"
    services.settings.update({"remote": {"hosts": {"box": {"token": "wrong"}}}})
    [status] = services.remote.status(refresh=True)
    assert status.state == "refused"
    job = services.tts.speak(SpeakRequest(model=MODEL, text="Hello.", voice_spec=VoiceInput(preset="alto")), wait=True)
    assert job.state == "failed" and job.error is not None and job.error.code == "remote_unavailable" and "refused" in job.error.message
    with pytest.raises(RemoteError):
        services.remote.info("box", refresh=True)
    with pytest.raises(DeviceError):
        services.settings.update({"engines": {"dummy": {"device": "nowhere/cuda:0"}}})
        services.runtime.device_for("dummy")


def test_speak_clone_and_voice_cache_run_on_the_agent(make_services, agent, wav_file):
    services = connect(make_services, agent)
    job = services.tts.speak(SpeakRequest(model=MODEL, text="Hello from afar.", voice_spec=VoiceInput(preset="alto"), params=CommonParams(seed=3)), wait=True)
    assert job.state == "completed", job.error
    out = services.outputs.get(job.result["output_id"])
    assert out.duration > 0.3
    status = services.runtime.status()
    [engine] = status.engines
    assert engine.device == "box/cpu" and engine.remote == "box"
    [model] = status.models
    assert (model.model_id, model.device, model.remote) == (MODEL, "box/cpu", "box") and model.bytes == 0
    assert [m.device for m in agent.services.runtime.status().models] == ["cpu"]

    # A cloned voice: the clip goes once, by content; the agent's voice cache comes back here.
    voice, _ = services.voices.create_from_audio("Tone", [wav_file], ["a tone"], consent="a test tone")
    req = SpeakRequest(model=MODEL, text="Cloned.", voice=voice.id)
    first = services.tts.speak(req, wait=True)
    assert first.state == "completed", first.error
    files = list((agent.files_dir).iterdir())
    assert files and all(len(f.name) == 64 for f in files)
    cache = list(services.voices.folder(voice).glob("cache/dummy/**/pitch.txt"))
    assert cache and cache[0].read_text(encoding="utf-8").strip()
    second = services.tts.speak(req, wait=True)
    assert second.state == "completed"
    assert len(list(agent.files_dir.iterdir())) == len(files)

    # A one-off reference and the same audio as the local engine makes.
    ref = SpeakRequest(model=MODEL, text="Again.", voice_spec=VoiceInput(references=[ReferenceInput(path=str(wav_file), transcript="a tone")]), params=CommonParams(seed=5))
    remote_out = services.tts.speak(ref, wait=True)
    assert remote_out.state == "completed", remote_out.error
    services.settings.update({"engines": {"dummy": {"device": "cpu"}}})
    local_out = services.tts.speak(ref, wait=True)
    from pathlib import Path

    from mizuoto.core.audio.io import decode

    a, _ = decode(Path(services.outputs.get(remote_out.result["output_id"]).path))
    b, _ = decode(Path(services.outputs.get(local_out.result["output_id"]).path))
    assert len(a) == len(b) and abs(a - b).max() < 1e-4


def test_streaming_and_cancel(make_services, agent):
    services = connect(make_services, agent)
    chunks: list[int] = []
    ends: list[list[str]] = []
    sink = StreamSink(chunk=lambda samples, rate, segment: chunks.append(len(samples)), end=ends.append)
    job = services.tts.speak(SpeakRequest(model=MODEL, text="One. Two. Three.", voice_spec=VoiceInput(preset="bass"), segmentation="mizuoto"), stream=sink)
    done = services.jobs.wait(job.id, 60)
    assert done.state == "completed", done.error
    out = services.outputs.get(done.result["output_id"])
    assert chunks and ends and ends[0] == done.result["output_ids"]
    assert abs(sum(chunks) / 24000 - out.duration) < 0.05

    slow = services.tts.speak(SpeakRequest(model=MODEL, text="A long one.", voice_spec=VoiceInput(preset="bass"), engine_options={"delay_ms": 3000}))
    deadline = time.monotonic() + 30
    while services.jobs.get(slow.id).state != "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)
    services.jobs.cancel(slow.id)
    assert services.jobs.wait(slow.id, 30).state == "cancelled"
    # The engine is still usable.
    again = services.tts.speak(SpeakRequest(model=MODEL, text="Still here.", voice_spec=VoiceInput(preset="bass")))
    assert services.jobs.wait(again.id, 60).state == "completed"


def test_lost_agent_fails_the_request_and_reconnects(make_services, agent, tmp_path):
    services = connect(make_services, agent, start_background=False)
    assert services.tts.speak(SpeakRequest(model=MODEL, text="Hi.", voice_spec=VoiceInput(preset="alto")), wait=True).state == "completed"
    sup = services.runtime.find("dummy")[0]
    errors: list[Exception] = []

    def speak() -> None:
        job = services.tts.speak(SpeakRequest(model=MODEL, text="Cut off.", voice_spec=VoiceInput(preset="alto"), engine_options={"delay_ms": 4000}), wait=True)
        errors.append(Exception(job.error.code if job.error else job.state))

    t = threading.Thread(target=speak)
    t.start()
    time.sleep(1.0)
    for s in list(agent._sessions):
        s.close()
    t.join(30)
    assert errors and str(errors[0]) == "remote_unavailable"
    assert sup.state in ("crashed", "stopped")
    job = services.tts.speak(SpeakRequest(model=MODEL, text="Back.", voice_spec=VoiceInput(preset="alto")), wait=True)
    assert job.state == "completed", job.error
    assert sup.info().restarts >= 1


def test_hosts_are_added_used_and_removed(make_services, agent):
    services = make_services()
    port = agent.address[1]
    with pytest.raises(Exception):
        services.remote.add_host("box", f"127.0.0.1:{port}")  # no token
    status = services.remote.add_host("box", f"127.0.0.1:{port}", "s3cret")
    assert status.state == "online"
    assert services.remote.add_host("box", f"127.0.0.1:{port}").state == "online"  # the token is kept
    assert services.remote.use("dummy", "box", "cpu") == "box/cpu"
    assert services.runtime.device_for("dummy") == "box/cpu" and services.engines.status("dummy").remote == "box"
    view = services.settings.view()
    assert view.remote.hosts["box"].token_configured and "s3cret" not in view.model_dump_json()
    assert services.remote.remove_host("box") == ["dummy"]
    assert services.remote.hosts() == {} and services.runtime.device_for("dummy") == "cpu"
    assert services.remote.status() == []


def test_api(make_services, agent):
    from fastapi.testclient import TestClient

    from mizuoto.api.app import create_app

    app = create_app(make_services(start_background=True), bound_host="127.0.0.1", serve_ui=False)
    origin = {"origin": "http://localhost"}
    with TestClient(app, base_url="http://localhost") as client:
        r = client.put("/api/v1/remote/hosts/box", json={"address": f"127.0.0.1:{agent.address[1]}", "token": "s3cret"}, headers=origin)
        assert r.status_code == 200 and r.json()["state"] == "online", r.text
        assert client.put("/api/v1/remote/engines/dummy", json={"host": "box", "device": "cpu"}, headers=origin).json() == {"device": "box/cpu"}
        job = client.post("/api/v1/tts/speak?wait=true", json={"model": MODEL, "text": "Through the API.", "voice_spec": {"preset": "tenor"}}, headers=origin).json()
        assert job["state"] == "completed", job
        hosts = client.get("/api/v1/remote/hosts").json()
        assert hosts[0]["engines"] == ["dummy"] and hosts[0]["info"]["name"]
        assert client.delete("/api/v1/remote/hosts/box", headers=origin).json() == ["dummy"]


def test_streamed_text_conversion_edit_and_ops_run_on_the_agent(make_services, agent, wav_file):
    from mizuoto.core.tts.models import ConvertRequest, EditRequest, EditSpanInput, SourceInput
    from mizuoto.core.tts.service import TextFeed

    services = connect(make_services, agent)
    feed = TextFeed()
    chunks: list[int] = []
    sink = StreamSink(chunk=lambda samples, rate, segment: chunks.append(len(samples)), text=feed)

    def write() -> None:
        for piece in ("Hello ", "there, ", "streamed ", "text."):
            time.sleep(0.05)
            feed.put(piece)
        feed.close()

    threading.Thread(target=write).start()
    job = services.tts.speak(SpeakRequest(model=MODEL, text="Well, ", text_stream=True, voice_spec=VoiceInput(preset="alto")), wait=True, stream=sink)
    assert job.state == "completed", job.error
    out = services.outputs.get(job.result["output_id"])
    assert out.text == "Well, Hello there, streamed text." and len(chunks) == 5  # one chunk per piece, as locally

    source = SourceInput(path=str(wav_file))
    job = services.tts.convert(ConvertRequest(model=MODEL, source=source, voice_spec=VoiceInput(references=[ReferenceInput(path=str(wav_file))])), wait=True)
    assert job.state == "completed", job.error
    out = services.outputs.get(job.result["output_id"])
    assert out.kind == "conversion" and abs(out.duration - 2.0) < 0.2
    job = services.tts.edit(EditRequest(model=MODEL, source=source, transcript="A new sentence.", edits=[EditSpanInput(start=0.5, end=1.0, text="new")]), wait=True)
    assert job.state == "completed", job.error
    job = services.tts.engine_op(MODEL, "echo", {"x": 1}, wait=True)
    assert job.state == "completed" and job.result["echo"] == {"x": 1}
    # A file the payload names crosses by content.
    from mizuoto.core.inference.engine import ModelRef

    handle = services.runtime.loaded(ModelRef(MODEL, "dummy", "default"), "box/cpu")
    assert handle.remote == "box" and handle.call("transcribe", {"path": str(wav_file)})["text"] == "dummy transcript of 2.0 seconds"
    assert services.runtime.engine("dummy", "box/cpu").status()["models"][0]["model_id"] == MODEL


def test_another_protocol_is_refused_with_an_upgrade_hint(make_services, agent):
    import socket

    from mizuoto.core.remote.frames import FrameConnection
    from mizuoto.core.remote.wire import AGENT_PROTOCOL, UPGRADE_HINT

    # A host of protocol 1 (Mizuoto 0.1) is refused by this agent.
    conn = FrameConnection(socket.create_connection(agent.address))
    conn.send({"type": "hello", "token": "s3cret", "protocol": 1})
    reply, _ = conn.recv()
    conn.close()
    assert reply["type"] == "error" and reply["code"] == "remote_unavailable" and reply["detail"]["hint"] == UPGRADE_HINT
    assert reply["detail"]["agent_protocol"] == AGENT_PROTOCOL == 2 and reply["detail"]["host_protocol"] == 1

    # An agent of protocol 1 refuses this host as Mizuoto 0.1 did; the host says how to fix it.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def old_agent() -> None:
        sock, _ = listener.accept()
        c = FrameConnection(sock)
        hello, _ = c.recv()
        c.send({"type": "error", "id": "", "code": "remote_unavailable", "message": f"The agent speaks protocol 1, the host {hello['protocol']}", "detail": {"agent": "0.1.0"}})
        c.close()

    threading.Thread(target=old_agent, daemon=True).start()
    services = make_services()
    services.settings.update({"remote": {"hosts": {"old": {"address": f"127.0.0.1:{listener.getsockname()[1]}", "token": "s3cret"}}}})
    with pytest.raises(RemoteError) as e:
        services.remote.info("old", refresh=True)
    listener.close()
    assert e.value.detail["refused"] and e.value.detail["hint"] == UPGRADE_HINT and "protocol 1" in e.value.message
