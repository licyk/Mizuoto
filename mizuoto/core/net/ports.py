"""Bind the listening socket ourselves, so the port that was tested is the port that serves.

Binding is the only reliable answer to "is this port free": a port that is merely unanswered may
still be reserved or blocked, and a connect-then-bind check leaves a gap another process can use.
"""

import errno
import logging
import socket

logger = logging.getLogger(__name__)

MAX_PORT_ATTEMPTS = 20


class PortUnavailableError(RuntimeError):
    pass


def _family(host: str) -> socket.AddressFamily:
    return socket.AF_INET6 if ":" in host else socket.AF_INET


def bind_first_free_port(host: str, start_port: int, attempts: int = MAX_PORT_ATTEMPTS, strict: bool = False) -> socket.socket:
    """Bind a listening socket to the first free port at or above ``start_port``.

    With ``strict``, only ``start_port`` is tried. Raises PortUnavailableError when nothing binds.
    """
    if start_port == 0:
        # Let the operating system pick any free port, which cannot collide.
        sock = socket.socket(_family(host), socket.SOCK_STREAM)
        sock.bind((host, 0))
        sock.listen(2048)
        sock.setblocking(False)
        return sock
    if not 1 <= start_port <= 65535:
        raise PortUnavailableError(f"Port {start_port} is out of range")
    last_port = start_port if strict else min(start_port + max(attempts, 1) - 1, 65535)
    last_error: OSError | None = None
    for port in range(start_port, last_port + 1):
        sock = socket.socket(_family(host), socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            # Windows: stop another process from binding the same address and port.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)  # type: ignore[attr-defined]
        else:
            # Elsewhere: allow a quick restart while old connections are in TIME_WAIT.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            sock.listen(2048)
        except OSError as e:
            sock.close()
            last_error = e
            # EACCES covers reserved ranges, such as Windows excluded ports.
            if e.errno in (errno.EADDRINUSE, errno.EACCES, getattr(errno, "WSAEADDRINUSE", -1), getattr(errno, "WSAEACCES", -1)):
                continue
            raise PortUnavailableError(f"Cannot bind {host}:{port}: {e}") from e
        sock.setblocking(False)
        return sock
    if strict:
        raise PortUnavailableError(f"Port {start_port} on {host} is unavailable ({last_error}), and strict port mode is on")
    raise PortUnavailableError(f"No free port in range {start_port}-{last_port} on {host}")


def is_loopback(host: str) -> bool:
    return host in ("localhost", "::1") or host.startswith("127.")
