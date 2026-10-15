"""The API over TestClient, against the dummy engine: errors, security, uploads, speech, voices, jobs."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from mizuoto.api.app import create_app

ORIGIN = {"origin": "http://localhost"}
MODEL = "dummy/default"


@pytest.fixture
def app(make_services):
    # As the server runs: the job scheduler and the worker reaper are on.
    return create_app(make_services(start_background=True), bound_host="127.0.0.1", serve_ui=False)


@pytest.fixture
def services(app):
    return app.state.services


@pytest.fixture
def client(app):
    with TestClient(app, base_url="http://localhost") as c:
        yield c


def upload(client, path, name="clip.wav"):
    r = client.put(f"/api/v1/audio/files?name={name}", content=path.read_bytes(), headers=ORIGIN)
    assert r.status_code == 200, r.text
    return r.json()


def test_health_meta_version(client):
    assert client.get("/api/v1/app/health").json() == {"status": "ok", "auth_required": False}
    meta = client.get("/api/v1/app/meta").json()
    assert meta["local"] is True and meta["first_run"] is True and "speak" in meta["features"]
    assert client.put("/api/v1/client-state/first_run_done", json=True, headers=ORIGIN).status_code == 200
    assert client.get("/api/v1/app/meta").json()["first_run"] is False
    assert client.get("/api/v1/app/version").json()["version"]


def test_error_shape(client):
    r = client.get("/api/v1/models/dummy/nope")
    assert r.status_code == 404 and r.json()["code"] == "not_found" and "nope" in r.json()["message"]
    r = client.post("/api/v1/tts/plan", json={"model": MODEL, "text": " "}, headers=ORIGIN)
    assert r.status_code == 400 and r.json()["code"] == "invalid_input"
    r = client.post("/api/v1/tts/plan", json={"model": "qwen3tts/1.7b-base", "text": "x"}, headers=ORIGIN)
    assert r.status_code == 409 and r.json()["code"] == "model_missing" and r.json()["detail"]["hint"]
    r = client.patch("/api/v1/settings", json={"tts": {"max_text_chars": 1}}, headers=ORIGIN)
    assert r.status_code == 400


def test_host_origin_and_token(app, services):
    with TestClient(app, base_url="http://evil.example") as c:
        assert c.get("/api/v1/app/health").status_code == 400
    with TestClient(app, base_url="http://localhost") as c:
        assert c.post("/api/v1/compute/release", headers={"origin": "http://evil.example"}).status_code == 403
        services.settings.update({"server": {"access_token": "s3cret"}})
        assert c.get("/api/v1/models").status_code == 401
        assert c.get("/api/v1/app/health").status_code == 200
        assert c.get("/api/v1/models", headers={"authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/api/v1/models?token=s3cret").status_code == 200


def test_listings(client):
    engines = {e["id"]: e for e in client.get("/api/v1/engines").json()}
    assert engines["dummy"]["available"] and engines["dummy"]["missing"] == [] and engines["asr"]["internal"]
    detail = client.get("/api/v1/engines/qwen3tts").json()
    assert "env" not in detail and detail["status"]["id"] == "qwen3tts" and detail["manifest"]["variants"]["base"]["capabilities"]
    assert {m["id"] for m in client.get("/api/v1/models?installed=true").json()} == {MODEL}
    assert client.get("/api/v1/models/qwen3tts/1.7b-base").json()["state"] == "missing"
    spec = client.get(f"/api/v1/tts/models/{MODEL}/spec").json()
    assert spec["sample_rate"] == 24000 and "preset" in spec["voice_modes"]
    assert client.get(f"/api/v1/models/{MODEL}/speakers").json() == ["alto", "bass", "soprano", "tenor"]
    assert client.get("/api/v1/envs").status_code == 404 and client.get("/api/v1/workers").status_code == 404
    runtime = client.get("/api/v1/runtime").json()
    assert runtime["models"] == [] and runtime["engines"] == []
    report = client.get("/api/v1/engines/dummy/doctor").json()
    assert report["install_command"] and [f["check"] for f in report["findings"] if f["engine"] == "dummy"] == ["engine"]
    assert client.get("/api/v1/engines/nope/doctor").status_code == 404
    meta = client.get("/api/v1/app/meta").json()
    assert meta["first_run"] and "torch" in meta
    assert client.get("/api/v1/compute/info").json()["devices"]


def test_engine_settings_and_licence(client, services):
    r = client.patch("/api/v1/engines/qwen3tts", json={"device": "cpu", "load_options": {"fused_projections": True}}, headers=ORIGIN)
    assert r.status_code == 200, r.text
    assert services.settings.settings.engines["qwen3tts"].load_options == {"fused_projections": True}
    assert client.patch("/api/v1/engines/qwen3tts", json={"load_options": {"nope": 1}}, headers=ORIGIN).json()["code"] == "parameter_not_supported"
    assert client.patch("/api/v1/engines/qwen3tts", json={"accepted_licence": "x"}, headers=ORIGIN).status_code == 400
    r = client.post("/api/v1/engines/dummy/licence", json={"licence": "gpl-3.0"}, headers=ORIGIN)
    assert r.status_code == 200 and r.json()["licence_accepted"]
    assert client.post("/api/v1/engines/dummy/licence", json={"licence": "gpl-3.0@old"}, headers=ORIGIN).status_code == 400


def test_upload_range_peaks_and_transcribe_errors(client, wav_file):
    data = wav_file.read_bytes()
    f = upload(client, wav_file, "voice.wav")
    assert f["sample_rate"] == 22050 and abs(f["duration"] - 2.0) < 0.01 and f["origin"] == "upload"
    r = client.get(f"/api/v1/audio/files/{f['id']}/content", headers={"range": "bytes=0-99"})
    assert r.status_code == 206 and r.content == data[:100]
    peaks = client.get(f"/api/v1/audio/files/{f['id']}/peaks?points=64").json()
    assert peaks["points"] == 64 and len(peaks["data"]) == 128
    r = client.put("/api/v1/audio/files?name=notes.txt", content=b"not audio", headers=ORIGIN)
    assert r.status_code == 400
    r = client.post("/api/v1/audio/transcribe", json={"audio_id": f["id"]}, headers=ORIGIN)
    assert r.status_code == 409 and r.json()["code"] == "engine_not_installed"


def test_browse_roots_are_enforced(client, services, tmp_path, wav_file):
    services.settings.update({"paths": {"browse_roots": [str(wav_file.parent)]}})
    listing = client.get("/api/v1/audio/browse", params={"path": str(wav_file.parent)}).json()
    assert [e["name"] for e in listing["entries"]] == ["voice.wav"]
    assert client.get("/api/v1/audio/browse", params={"path": str(tmp_path)}).status_code == 400
    r = client.post("/api/v1/audio/files/paths", json={"paths": [str(wav_file)]}, headers=ORIGIN)
    assert r.status_code == 200 and r.json()[0]["origin"] == "path"
    r = client.post("/api/v1/models/import", json={"path": str(tmp_path)}, headers=ORIGIN)
    assert r.status_code == 400 and r.json()["code"] == "invalid_path"


def test_speak_plan_wait_outputs_and_rerun(client, services):
    req = {"model": MODEL, "text": "Hello there.", "voice_spec": {"preset": "alto"}, "params": {"seed": 4}}
    plan = client.post("/api/v1/tts/plan", json=req, headers=ORIGIN).json()
    assert plan["voice_mode"] == "preset" and plan["segments"] == ["Hello there."]
    job = client.post("/api/v1/tts/speak?wait=true", json=req, headers=ORIGIN).json()
    assert job["state"] == "completed", job
    out_id = job["result"]["output_id"]
    out = client.get(f"/api/v1/outputs/{out_id}").json()
    assert out["model"] == MODEL and out["params"]["seed"] == 4
    audio = client.get(f"/api/v1/outputs/{out_id}/content")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    assert len(client.get(f"/api/v1/outputs/{out_id}/peaks?points=32").json()["data"]) == 64
    rerun = client.get(f"/api/v1/outputs/{out_id}/rerun").json()
    assert rerun["voice_spec"]["preset"] == "alto" and rerun["params"]["seed"] == 4
    page = client.get("/api/v1/outputs?q=Hello").json()
    assert [o["id"] for o in page["items"]] == [out_id]
    assert client.patch(f"/api/v1/outputs/{out_id}", json={"favourite": True}, headers=ORIGIN).json()["favourite"]
    z = client.post("/api/v1/outputs/zip", json={"ids": [out_id]}, headers=ORIGIN)
    assert z.status_code == 200 and z.content[:2] == b"PK"
    assert client.post("/api/v1/outputs/delete", json={"ids": [out_id]}, headers=ORIGIN).json() == [out_id]
    seg = client.post("/api/v1/tts/segment", json={"text": "One. " * 80, "model": MODEL}, headers=ORIGIN).json()
    assert seg["segmented_by"] == "mizuoto" and len(seg["segments"]) > 1
    runtime = client.get("/api/v1/runtime").json()
    assert [m["model_id"] for m in runtime["models"]] == [MODEL] and runtime["models"][0]["device"] == "cpu"
    assert client.post("/api/v1/runtime/unload", json={"model": "dummy/other"}, headers=ORIGIN).status_code == 404
    assert client.post("/api/v1/runtime/unload", json={"model": MODEL}, headers=ORIGIN).json()["models"] == []


def test_speak_job_runs_in_the_background(app):
    with TestClient(app, base_url="http://localhost") as c:
        job = c.post("/api/v1/tts/speak", json={"model": MODEL, "text": "Background."}, headers=ORIGIN).json()
        assert job["state"] in ("queued", "running", "completed")
        deadline = time.monotonic() + 30
        while c.get(f"/api/v1/jobs/{job['id']}").json()["state"] != "completed":
            assert time.monotonic() < deadline
            time.sleep(0.1)
        assert c.get(f"/api/v1/jobs/{job['id']}/log").status_code == 200
        assert any(j["id"] == job["id"] for j in c.get("/api/v1/jobs").json()["items"])


def test_voices_over_api(client, wav_file, tmp_path):
    f = upload(client, wav_file)
    r = client.post("/api/v1/voices", json={"name": "Web Voice", "audio_ids": [f["id"]], "transcripts": ["Hello."], "language": "en"}, headers=ORIGIN)
    assert r.status_code == 200, r.text
    voice = r.json()["voice"]
    assert voice["slug"] == "web-voice" and voice["references"][0]["transcript"] == "Hello."
    assert [v["slug"] for v in client.get("/api/v1/voices").json()] == ["web-voice"]
    fit = client.get(f"/api/v1/voices/{voice['id']}/fit?model={MODEL}").json()
    assert fit["usable"] and fit["mode"] == "clone"
    assert [x["model"] for x in client.get(f"/api/v1/voices/{voice['id']}/fits").json()] == [MODEL]
    assert [x["voice_id"] for x in client.get(f"/api/v1/tts/models/{MODEL}/voice-fits").json()] == [voice["id"]]
    sha = voice["references"][0]["sha256"]
    content = client.get(f"/api/v1/voices/{voice['id']}/references/{sha[:12]}/content")
    assert content.status_code == 200 and content.headers["content-type"] == "audio/flac"
    assert client.get(f"/api/v1/voices/{voice['id']}/references/{sha}/peaks?points=16").json()["points"] == 16
    voice = client.patch(f"/api/v1/voices/{voice['id']}/references/{sha}", json={"transcript": "Hi."}, headers=ORIGIN).json()
    assert voice["references"][0]["transcript"] == "Hi."
    assert client.patch(f"/api/v1/voices/{voice['id']}", json={"tags": ["a"]}, headers=ORIGIN).json()["tags"] == ["a"]
    job = client.post("/api/v1/tts/speak?wait=true", json={"model": MODEL, "text": "Library voice.", "voice": voice["slug"]}, headers=ORIGIN).json()
    assert job["state"] == "completed"
    saved = client.post("/api/v1/voices/from-output", json={"output_id": job["result"]["output_id"], "name": "Kept"}, headers=ORIGIN).json()
    assert saved["kind"] == "reference"
    preset = client.post("/api/v1/voices/preset", json={"model": MODEL, "speaker": "bass"}, headers=ORIGIN).json()
    assert preset["kind"] == "preset"
    archive = client.get(f"/api/v1/voices/{voice['id']}/export")
    assert archive.status_code == 200 and archive.content[:2] == b"PK"
    imported = client.put("/api/v1/voices/import?name=Again", content=archive.content, headers=ORIGIN)
    assert imported.status_code == 200, imported.text
    assert imported.json()["slug"] == "again"
    assert client.delete("/api/v1/voices/again", headers=ORIGIN).status_code == 204
    assert client.get("/api/v1/voices/again").status_code == 404


def test_openapi_lists_events_and_operations(client):
    spec = client.get("/openapi.json").json()
    events = spec["components"]["schemas"]["ServerEvents"]["properties"]
    assert {"job_updated", "voices_changed", "models_changed", "engines_changed", "runtime_changed"} <= set(events) and "workers_changed" not in events
    ops = {op["operationId"] for path in spec["paths"].values() for op in path.values()}
    assert {"speak", "plan_speech", "create_voice", "download_models", "get_runtime", "unload_models", "get_doctor", "list_jobs"} <= ops
    assert "install_env" not in ops
    # Responses list defaulted fields as required; a model also used as a request body splits in two.
    job = spec["components"]["schemas"]["Job"]
    assert {"steps", "can_retry"} <= set(job["required"])


def test_stream_websocket(client, services):
    with client.websocket_connect("ws://localhost/api/v1/tts/stream", headers=ORIGIN) as ws:
        ws.send_text(json.dumps({"model": MODEL, "texts": ["One.", "Two.", "Three."]}))
        assert ws.receive_json()["type"] == "queued"
        start = ws.receive_json()
        assert start["type"] == "start" and start["sample_rate"] == 24000 and start["granularity"] == "segment"
        segments, audio = [], 0
        while True:
            message = ws.receive()
            if message.get("bytes") is not None:
                audio += len(message["bytes"])
                continue
            data = json.loads(message["text"])
            if data["type"] == "segment":
                segments.append(data["index"])
            else:
                break
        assert data["type"] == "end" and segments == [0, 1, 2] and audio > 24000 and len(data["output_ids"]) == 3
        assert services.outputs.get(data["output_id"]).text == "One."


def test_stream_websocket_text_input_and_errors(client, services):
    with client.websocket_connect("ws://localhost/api/v1/tts/stream", headers=ORIGIN) as ws:
        ws.send_text(json.dumps({"model": MODEL, "text": "", "text_stream": True}))
        assert ws.receive_json()["type"] == "queued"
        for piece in ("Streamed ", "as it ", "is written."):
            ws.send_text(json.dumps({"type": "text", "text": piece}))
        ws.send_text(json.dumps({"type": "flush"}))
        while True:
            message = ws.receive()
            if message.get("text") and json.loads(message["text"])["type"] in ("end", "error"):
                data = json.loads(message["text"])
                break
        assert data["type"] == "end", data
        assert services.outputs.get(data["output_id"]).text == "Streamed as it is written."
    with client.websocket_connect("ws://localhost/api/v1/tts/stream", headers=ORIGIN) as ws:
        ws.send_text(json.dumps({"model": MODEL, "text": " "}))
        assert ws.receive_json()["code"] == "invalid_input"
    with client.websocket_connect("ws://localhost/api/v1/tts/stream", headers=ORIGIN) as ws:
        ws.send_text("not json")
        assert ws.receive_json()["code"] == "invalid_input"
    with client.websocket_connect("ws://localhost/api/v1/tts/stream", headers=ORIGIN) as ws:
        ws.send_text(json.dumps({"model": MODEL, "text": "Slow.", "engine_options": {"delay_ms": 400, "chunk_ms": 50}}))
        job_id = ws.receive_json()["job_id"]
        ws.receive_json()
        ws.send_text(json.dumps({"type": "cancel"}))
        while True:
            message = ws.receive()
            if message.get("text"):
                data = json.loads(message["text"])
                if data["type"] in ("end", "error"):
                    break
        assert data["code"] == "cancelled"
    assert services.jobs.wait(job_id, 10).state == "cancelled"


def test_speak_streams_over_http(client):
    r = client.post("/api/v1/tts/speak?stream=true", json={"model": MODEL, "text": "Hello there."}, headers=ORIGIN)
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.headers["x-mizuoto-sample-rate"] == "24000"
    assert r.content[:4] == b"RIFF" and len(r.content) > 44 + 24000
    r = client.post("/api/v1/tts/speak?stream=true&stream_format=pcm", json={"model": MODEL, "text": "Hi."}, headers=ORIGIN)
    assert r.headers["content-type"] == "audio/pcm" and r.content[:4] != b"RIFF" and len(r.content) % 2 == 0
    r = client.post("/api/v1/tts/speak?stream=true", json={"model": MODEL, "text": "x", "text_stream": True}, headers=ORIGIN)
    assert r.status_code == 400


def test_openai_compatible_routes(client, services):
    r = client.post("/v1/audio/speech", json={"model": MODEL, "input": "Hello there.", "voice": "alto", "response_format": "wav"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF" and r.headers["x-mizuoto-model"] == MODEL
    r = client.post("/v1/audio/speech", json={"model": MODEL, "input": "Hi.", "voice": "alloy", "response_format": "pcm"})
    assert r.headers["content-type"] == "audio/pcm" and r.content[:4] != b"RIFF"
    r = client.post("/v1/audio/speech", json={"model": MODEL, "input": "Hi.", "response_format": "pcm", "stream_format": "audio"})
    assert r.status_code == 200 and r.headers["x-mizuoto-sample-rate"] == "24000"
    r = client.post("/v1/audio/speech", json={"model": "tts-1", "input": "Hi.", "voice": "alloy"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    assert MODEL in [m["id"] for m in client.get("/v1/models").json()["data"]]
    services.settings.update({"server": {"access_token": "s3cret"}})
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"authorization": "Bearer s3cret"}).status_code == 200


def test_presets_over_api(client):
    r = client.post("/api/v1/presets", json={"name": "Soft", "model": MODEL, "params": {"temperature": 0.4}}, headers=ORIGIN)
    assert r.status_code == 200, r.text
    preset = r.json()
    assert client.post(f"/api/v1/presets/{preset['id']}/default", json={"default": True}, headers=ORIGIN).json()["is_default"] is True
    assert [p["name"] for p in client.get(f"/api/v1/presets?model={MODEL}").json()] == ["Soft"]
    assert client.patch(f"/api/v1/presets/{preset['id']}", json={"name": "Softer"}, headers=ORIGIN).json()["name"] == "Softer"
    assert client.post("/api/v1/presets", json={"name": "Bad", "model": MODEL, "params": {"top_k": 2}}, headers=ORIGIN).json()["code"] == "parameter_not_supported"
    assert client.delete(f"/api/v1/presets/{preset['id']}", headers=ORIGIN).status_code == 200
    assert client.get("/api/v1/presets").json() == []


def test_datasets_and_training(client, services, tmp_path):
    from tests.core.test_training import tone

    r = client.post("/api/v1/datasets", json={"name": "Tones", "language": "en"}, headers=ORIGIN)
    assert r.status_code == 200, r.text
    ds = r.json()
    for i in range(3):
        clip = tone(tmp_path / f"c{i}.wav", 3.0 + i)
        r = client.put(f"/api/v1/datasets/{ds['id']}/files?name=c{i}.wav&text=Clip {i}.", content=clip.read_bytes(), headers=ORIGIN)
        assert r.status_code == 200, r.text
    assert r.json()["dataset"]["stats"]["rows"] == 3 and not list((services.settings.data_dir / "uploads" / "incoming").iterdir())
    rows = client.get(f"/api/v1/datasets/{ds['id']}/rows").json()
    assert rows["total"] == 3 and rows["items"][0]["text"].startswith("Clip")
    row = rows["items"][0]
    assert client.patch(f"/api/v1/datasets/{ds['id']}/rows/{row['id']}", json={"text": "Changed."}, headers=ORIGIN).json()["text"] == "Changed."
    audio = client.get(f"/api/v1/datasets/{ds['id']}/rows/{row['id']}/audio")
    assert audio.status_code == 200 and audio.content[:4] == b"fLaC"
    assert client.get(f"/api/v1/datasets/{ds['id']}/report").json()["usable"] == 3
    job = client.post(f"/api/v1/datasets/{ds['id']}/prepare", json={"transcribe": False}, headers=ORIGIN).json()
    assert services.jobs.wait(job["id"], 30).state == "completed"

    methods = client.get("/api/v1/train/methods?model=dummy/default").json()
    assert [m["id"] for m in methods] == ["speaker", "full"] and all(m["available"] for m in methods)
    r = client.post("/api/v1/train/experiments", json={"name": "t1", "base_model": "dummy/default", "dataset": ds["id"], "hyperparams": {"epochs": 1}}, headers=ORIGIN)
    assert r.status_code == 200, r.text
    assert r.json()["method"] == "speaker"
    job = client.post("/api/v1/train/experiments/dummy/t1/run", json={}, headers=ORIGIN).json()
    assert services.jobs.wait(job["id"], 60).state == "completed"
    exp = client.get("/api/v1/train/experiments/dummy/t1").json()
    assert [s["status"] for s in exp["stages"]] == ["done"] * 3 and exp["artifacts"][0]["kind"] == "voice"
    assert client.get("/api/v1/train/experiments/dummy/t1/metrics?stage=train").json()
    assert client.post("/api/v1/train/experiments", json={"name": "t1", "base_model": "dummy/default", "dataset": ds["id"]}, headers=ORIGIN).status_code == 409
    assert client.delete("/api/v1/train/experiments/dummy/t1", headers=ORIGIN).status_code == 200
    assert client.delete(f"/api/v1/datasets/{ds['id']}", headers=ORIGIN).status_code == 200
    assert client.get("/api/v1/datasets").json() == []


def test_projects(client, services):
    body = {"name": "Dialogue", "model": MODEL, "speakers": [{"id": "narrator", "name": "Narrator", "preset": "alto"}]}
    r = client.post("/api/v1/projects", json=body, headers=ORIGIN)
    assert r.status_code == 200, r.text
    p = r.json()
    p = client.post(f"/api/v1/projects/{p['id']}/script", json={"text": "# One\nHello.\nAda: Hi.\nBo: Hey.\nAda: Bye."}, headers=ORIGIN).json()
    assert [s["name"] for s in p["speakers"]] == ["Narrator", "Ada", "Bo"] and p["stats"]["missing"] == 4
    assert client.get(f"/api/v1/projects/{p['id']}/script").text.startswith("# One\n\nHello.")
    line = p["chapters"][0]["lines"][1]
    req = client.get(f"/api/v1/projects/{p['id']}/lines/{line['id']}/request").json()
    assert req["text"] == "Hi." and req["model"] == MODEL and req["output"]["project"] == p["id"]
    job = client.post(f"/api/v1/projects/{p['id']}/render", json={}, headers=ORIGIN).json()
    assert services.jobs.wait(job["id"], 60).state == "completed"
    p = client.get(f"/api/v1/projects/{p['id']}").json()
    assert p["stats"]["rendered"] == 4 and [s["id"] for s in client.get("/api/v1/projects").json()] == [p["id"]]
    assert all(o["kind"] != "line" for o in client.get("/api/v1/outputs").json()["items"])
    # The editor's PUT: a line's text changes, its takes stay.
    edit = {k: p[k] for k in ("name", "model", "speakers", "chapters")}
    edit["chapters"][0]["lines"][0]["text"] = "Hello again."
    p = client.put(f"/api/v1/projects/{p['id']}", json=edit, headers=ORIGIN).json()
    assert p["chapters"][0]["lines"][0]["state"] == "stale" and p["chapters"][0]["lines"][0]["takes"]
    first = p["chapters"][0]["lines"][0]["takes"][0]["output_id"]
    assert client.put(f"/api/v1/projects/{p['id']}/lines/{line['id']}/take", json={"output_id": first}, headers=ORIGIN).status_code == 404
    assert client.post(f"/api/v1/projects/{p['id']}/export", json={}, headers=ORIGIN).status_code == 400
    job = client.post(f"/api/v1/projects/{p['id']}/export", json={"allow_missing": True, "format": "flac"}, headers=ORIGIN).json()
    done = services.jobs.wait(job["id"], 60)
    assert done.state == "completed", done.error
    out = done.result["output_id"]
    srt = client.get(f"/api/v1/projects/exports/{out}/subtitles")
    assert srt.status_code == 200 and "Hi." in srt.text and srt.headers["content-type"].startswith("application/x-subrip")
    assert client.get(f"/api/v1/projects/exports/{out}/subtitles?format=vtt").text.startswith("WEBVTT")
    assert client.delete(f"/api/v1/projects/{p['id']}", headers=ORIGIN).status_code == 204
    assert client.get(f"/api/v1/projects/{p['id']}").status_code == 404
