"""Frames (the remote agent), tasks, and job-worker events (JSON lines)."""

import socket
import struct
import threading

import pytest

from mizuoto.core.inference.tasks import EmotionSpec, Reference, StyleSpec, SynthesisTask, VoiceSpec, synthesis_task_from_dict, to_dict
from mizuoto.core.jobs.events import ErrorEvent, LogEvent, ProgressEvent
from mizuoto.core.jobs.jsonl import PREFIX, Emitter, parse_line
from mizuoto.core.remote.frames import MAX_FRAME, ConnectionClosed, FrameConnection, FrameError, decode_body, encode_frame


def test_frames_round_trip_json_and_binary():
    a, b = socket.socketpair()
    ca, cb = FrameConnection(a), FrameConnection(b)
    payload = bytes(range(256)) * 4096  # 1 MiB, more than a socket buffer: read in several recv calls

    def send():
        ca.send({"type": "x", "text": "你好"})
        ca.send({"type": "chunk", "n": 1}, payload)

    sender = threading.Thread(target=send)
    sender.start()
    assert cb.recv() == ({"type": "x", "text": "你好"}, None)
    msg, data = cb.recv()
    assert msg == {"type": "chunk", "n": 1} and data == payload
    sender.join()
    ca.close()
    with pytest.raises(ConnectionClosed):
        cb.recv()
    cb.close()


def test_frames_from_several_threads_do_not_interleave():
    a, b = socket.socketpair()
    ca, cb = FrameConnection(a), FrameConnection(b)
    blob = b"\x01" * 200_000

    def send(i):
        for _ in range(20):
            ca.send({"i": i}, blob)

    threads = [threading.Thread(target=send, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    got = [cb.recv() for _ in range(80)]
    for t in threads:
        t.join()
    assert all(len(p or b"") == len(blob) for _, p in got)
    assert sorted(m["i"] for m, _ in got) == sorted(list(range(4)) * 20)


def test_bad_frames():
    with pytest.raises(FrameError):
        decode_body(b"Xabc")
    with pytest.raises(FrameError):
        decode_body(b"J[1,2]")
    a, b = socket.socketpair()
    a.sendall(struct.pack(">I", MAX_FRAME + 1))
    with pytest.raises(FrameError):
        FrameConnection(b).recv()
    assert encode_frame({"a": 1})[4:5] == b"J"


def test_tasks_round_trip():
    task = SynthesisTask(
        text="hi",
        language="en",
        voice=VoiceSpec(references=[Reference(audio_path="/a.wav", transcript="hello", sha256="ab")]),
        style=StyleSpec(instruct="softly", emotion=EmotionSpec(vector={"happy": 0.5})),
        params={"seed": 1},
    )
    back = synthesis_task_from_dict(to_dict(task))
    assert back == task
    assert isinstance(back.voice.references[0], Reference)


def test_jsonl_events_and_noise():
    lines: list[str] = []

    class Stream:
        def write(self, s):
            lines.append(s)

        def flush(self):
            pass

    Emitter(Stream()).emit(ProgressEvent(value=0.5, step="s"))  # ty: ignore[invalid-argument-type]
    assert lines[0].startswith(PREFIX)
    assert parse_line(lines[0]) == ProgressEvent(value=0.5, step="s")
    assert parse_line("some library print") == LogEvent(level="info", message="some library print")
    assert parse_line(PREFIX + "{not json") == LogEvent(level="info", message=PREFIX + "{not json")
    err = parse_line(PREFIX + '{"type": "error", "code": "x", "message": "m", "extra": 1}')
    assert err == ErrorEvent(code="x", message="m")


def test_an_event_after_an_unfinished_line_is_still_an_event():
    from mizuoto.core.jobs.events import LogEvent, MetricEvent
    from mizuoto.core.jobs.jsonl import PREFIX, parse_lines

    glued = "Epoch 1/1:  50%|#####| 1/2" + PREFIX + '{"type": "metric", "epoch": 1, "step": 1, "values": {"loss": 0.5}}\n'
    events = parse_lines(glued)
    assert isinstance(events[0], LogEvent) and events[0].message.startswith("Epoch")
    assert isinstance(events[1], MetricEvent) and events[1].values == {"loss": 0.5}
    assert [type(e) for e in parse_lines("plain text\n")] == [LogEvent]
