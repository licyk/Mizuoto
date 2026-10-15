"""Length-prefixed frames over a stream socket: a JSON object, optionally followed by binary data.

A frame is a 4-byte big-endian length, then a 1-byte kind, then the body:

- ``J``: the body is one UTF-8 JSON object.
- ``B``: the body is a 4-byte big-endian header length, the JSON header, then raw bytes (audio:
  float32 little-endian mono).

The host ⇄ agent connection (``wire.py``) speaks these. ``multiprocessing.connection`` is not used: its
authentication handshake differs across Python versions, and a host and its agents may run different ones (3.10 to 3.13).
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from typing import Any

MAX_FRAME = 512 * 1024 * 1024
"""Frames above this are refused: a corrupt length would otherwise allocate gigabytes."""

KIND_JSON = b"J"
KIND_BINARY = b"B"


class ConnectionClosed(Exception):
    """The other end closed the connection (or it broke)."""


class FrameError(Exception):
    """A frame that does not parse."""


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        try:
            chunk = sock.recv(min(remaining, 4 * 1024 * 1024))
        except OSError as e:
            raise ConnectionClosed(str(e)) from e
        if not chunk:
            raise ConnectionClosed("connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def encode_frame(message: dict[str, Any], payload: bytes | None = None) -> bytes:
    """One frame holding ``message`` and, when given, ``payload``."""
    header = json.dumps(message, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    if payload is None:
        body = KIND_JSON + header
    else:
        body = KIND_BINARY + struct.pack(">I", len(header)) + header + payload
    if len(body) > MAX_FRAME:
        raise FrameError(f"frame of {len(body)} bytes is too large")
    return struct.pack(">I", len(body)) + body


def decode_body(body: bytes) -> tuple[dict[str, Any], bytes | None]:
    """The message and payload of a frame body (without its length prefix)."""
    if not body:
        raise FrameError("empty frame")
    kind, rest = body[:1], body[1:]
    try:
        if kind == KIND_JSON:
            message, payload = json.loads(rest.decode("utf-8")), None
        elif kind == KIND_BINARY:
            if len(rest) < 4:
                raise FrameError("binary frame without a header length")
            (size,) = struct.unpack(">I", rest[:4])
            message, payload = json.loads(rest[4 : 4 + size].decode("utf-8")), rest[4 + size :]
        else:
            raise FrameError(f"unknown frame kind {kind!r}")
    except (ValueError, UnicodeDecodeError) as e:
        raise FrameError(f"frame does not parse: {e}") from e
    if not isinstance(message, dict):
        raise FrameError("a frame must hold a JSON object")
    return message, payload


class FrameConnection:
    """Frames over a connected socket. ``send`` is safe from several threads; ``recv`` is for one reader."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self._send_lock = threading.Lock()

    def send(self, message: dict[str, Any], payload: bytes | None = None) -> None:
        data = encode_frame(message, payload)
        with self._send_lock:
            try:
                self.sock.sendall(data)
            except OSError as e:
                raise ConnectionClosed(str(e)) from e

    def recv(self) -> tuple[dict[str, Any], bytes | None]:
        (size,) = struct.unpack(">I", _recv_exact(self.sock, 4))
        if size > MAX_FRAME:
            raise FrameError(f"frame of {size} bytes is too large")
        return decode_body(_recv_exact(self.sock, size))

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
