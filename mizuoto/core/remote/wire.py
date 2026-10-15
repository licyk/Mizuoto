"""The host ⇄ agent connection: frames (``mizuoto.core.remote.frames``) over TCP, or TLS.

Files cross by content: a path in a request becomes ``mizuoto-file:<sha256>/<name>`` once the agent
has the file (``have`` / ``put``), and the folder a ``prepare_voice`` writes into is ``mizuoto-out:out``,
whose files come back with the result.

``AGENT_PROTOCOL`` 2 (Mizuoto 0.2) works at the level of a loaded engine (``core/remote/agent.py``); protocol 1
(Mizuoto 0.1) relayed engine-worker messages. A host and an agent of different protocols refuse each other at
``hello``.
"""

from __future__ import annotations

import socket
import ssl
from pathlib import Path

from mizuoto.core.settings.models import AGENT_PORT

AGENT_PROTOCOL = 2
UPGRADE_HINT = "Install the same Mizuoto version on both machines (pip install -U mizuoto): a host and its agents must speak the same protocol"
FILE_REF = "mizuoto-file:"
OUT_REF = "mizuoto-out:"
CHUNK = 8 * 1024 * 1024
"""Bytes per ``put`` frame."""


def parse_address(address: str, default_port: int = AGENT_PORT) -> tuple[str, int]:
    """``host``, ``host:port``, ``[v6]:port`` → (host, port)."""
    text = address.strip()
    for prefix in ("tcp://", "mizuoto://"):
        text = text.removeprefix(prefix)
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        return host, int(rest.lstrip(":") or default_port)
    if text.count(":") == 1:
        host, port = text.split(":")
        return host, int(port)
    return text, default_port


def client_socket(address: str, tls: bool, ca_file: str, timeout: float) -> socket.socket:
    host, port = parse_address(address)
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    if tls:
        context = ssl.create_default_context(cafile=str(Path(ca_file).expanduser()) if ca_file else None)
        if ca_file:
            # A self-signed agent certificate given as its own CA names no host we can check.
            context.check_hostname = False
        sock = context.wrap_socket(sock, server_hostname=host)
    sock.settimeout(None)
    return sock


def server_context(certfile: str, keyfile: str | None) -> ssl.SSLContext:
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(str(Path(certfile).expanduser()), str(Path(keyfile).expanduser()) if keyfile else None)
    return context
