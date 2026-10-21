"""Bind the listening socket ourselves, so the port that was tested is the port that serves.

Binding is the only reliable answer to "is this port free": a port that is merely unanswered may
still be reserved or blocked, and a connect-then-bind check leaves a gap another process can use.
"""

import errno
import logging
import socket
from collections.abc import Iterator

logger = logging.getLogger(__name__)

MAX_PORT = 65535
# Searching wraps around to the ports below the requested one, but never into the privileged ones.
LOWEST_FALLBACK_PORT = 1024

# The port is taken or reserved (EACCES covers Windows excluded port ranges), so the next one may work.
# Any other error, such as an address this machine does not have, would fail on every port.
_TRY_NEXT_ERRNOS = frozenset(code for code in (errno.EADDRINUSE, errno.EACCES, getattr(errno, "WSAEADDRINUSE", None), getattr(errno, "WSAEACCES", None)) if code is not None)


class PortUnavailableError(RuntimeError):
    pass


def _family(host: str) -> socket.AddressFamily:
    return socket.AF_INET6 if ":" in host else socket.AF_INET


def _socket(host: str) -> socket.socket:
    """A TCP socket for ``host``. On the IPv6 wildcard (``::``) it also takes IPv4 connections: ``IPV6_V6ONLY`` is off by
    default on Linux but on by default on Windows, so it is turned off explicitly (as ``socket.create_server``'s
    ``dualstack_ipv6`` does)."""
    sock = socket.socket(_family(host), socket.SOCK_STREAM)
    if host.strip("[]") == "::" and hasattr(socket, "IPV6_V6ONLY"):
        try:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError:
            pass  # a system without dual-stack sockets: IPv6 only
    return sock


def _candidate_ports(start_port: int) -> Iterator[int]:
    """Every port from ``start_port`` up to the last one, then the unprivileged ports below ``start_port``."""
    yield from range(start_port, MAX_PORT + 1)
    yield from range(LOWEST_FALLBACK_PORT, start_port)


def _bind(host: str, port: int) -> socket.socket:
    """A listening socket on ``host:port``. Raises OSError when it cannot bind."""
    sock = _socket(host)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            # Windows: stop another process from binding the same address and port.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)  # type: ignore[attr-defined]
        else:
            # Elsewhere: allow a quick restart while old connections are in TIME_WAIT.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(2048)
    except OSError:
        sock.close()
        raise
    sock.setblocking(False)
    return sock


def bind_first_free_port(host: str, start_port: int, strict: bool = False) -> socket.socket:
    """Bind a listening socket to the first free port at or above ``start_port``.

    The search has no attempt limit: it runs up to port 65535, then wraps around to the unprivileged ports
    below ``start_port``. With ``strict``, only ``start_port`` is tried. Raises PortUnavailableError when
    nothing binds, or at once on an error that another port would not fix.
    """
    if start_port == 0:
        # Let the operating system pick any free port, which cannot collide.
        sock = _socket(host)
        sock.bind((host, 0))
        sock.listen(2048)
        sock.setblocking(False)
        return sock
    if not 1 <= start_port <= MAX_PORT:
        raise PortUnavailableError(f"Port {start_port} is out of range")
    candidates = (start_port,) if strict else _candidate_ports(start_port)
    tried = 0
    last_error: OSError | None = None
    for port in candidates:
        tried += 1
        try:
            return _bind(host, port)
        except OSError as e:
            if e.errno not in _TRY_NEXT_ERRNOS:
                raise PortUnavailableError(f"Cannot bind {host}:{port}: {e}") from e
            last_error = e
    if strict:
        raise PortUnavailableError(f"Port {start_port} on {host} is unavailable ({last_error}), and strict port mode is on")
    raise PortUnavailableError(f"No free port on {host}: none of the {tried} ports searched from {start_port} (wrapping around at {MAX_PORT}) could be bound ({last_error})")


def is_loopback(host: str) -> bool:
    return host in ("localhost", "::1") or host.startswith("127.")
