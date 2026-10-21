from __future__ import annotations

import errno
import socket

import pytest

from mizuoto.core.net import ports
from mizuoto.core.net.ports import PortUnavailableError, bind_first_free_port, is_loopback


def _ipv6() -> bool:
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


def test_a_taken_port_moves_to_the_next_unless_strict():
    first = bind_first_free_port("127.0.0.1", 0)
    try:
        port = first.getsockname()[1]
        if port >= 65535:
            pytest.skip("the system gave the last port")
        nxt = bind_first_free_port("127.0.0.1", port)
        try:
            assert nxt.getsockname()[1] > port
        finally:
            nxt.close()
        with pytest.raises(PortUnavailableError):
            bind_first_free_port("127.0.0.1", port, strict=True)
    finally:
        first.close()


def _taken(port: int) -> OSError:
    return OSError(errno.EADDRINUSE, f"port {port} is taken")


def test_search_has_no_attempt_limit(monkeypatch):
    # Far more taken ports than the old cap of 20.
    tried = []

    def fake_bind(host, port):
        tried.append(port)
        if port < 8000 + 500:
            raise _taken(port)
        return port

    monkeypatch.setattr(ports, "_bind", fake_bind)
    assert bind_first_free_port("127.0.0.1", 8000) == 8500
    assert tried == list(range(8000, 8501))


def test_search_wraps_around_below_the_start(monkeypatch):
    def fake_bind(host, port):
        if port >= 65000:
            raise _taken(port)
        return port

    monkeypatch.setattr(ports, "_bind", fake_bind)
    assert bind_first_free_port("127.0.0.1", 65000) == ports.LOWEST_FALLBACK_PORT


def test_every_port_taken_fails_after_trying_each_once(monkeypatch):
    tried = []

    def fake_bind(host, port):
        tried.append(port)
        raise _taken(port)

    monkeypatch.setattr(ports, "_bind", fake_bind)
    with pytest.raises(PortUnavailableError, match="No free port"):
        bind_first_free_port("127.0.0.1", 8000)
    assert sorted(tried) == list(range(ports.LOWEST_FALLBACK_PORT, ports.MAX_PORT + 1))


def test_an_error_another_port_would_not_fix_stops_the_search(monkeypatch):
    tried = []

    def fake_bind(host, port):
        tried.append(port)
        raise OSError(errno.EADDRNOTAVAIL, "address not available")

    monkeypatch.setattr(ports, "_bind", fake_bind)
    with pytest.raises(PortUnavailableError, match="Cannot bind"):
        bind_first_free_port("127.0.0.1", 8000)
    assert tried == [8000]


def test_out_of_range_ports_are_refused():
    with pytest.raises(PortUnavailableError):
        bind_first_free_port("127.0.0.1", 70000)


@pytest.mark.skipif(not _ipv6(), reason="no IPv6 on this machine")
def test_the_ipv6_wildcard_also_takes_ipv4_connections():
    server = bind_first_free_port("::", 0)
    try:
        server.setblocking(True)
        port = server.getsockname()[1]
        assert server.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 0
        for family, address in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
            with socket.socket(family, socket.SOCK_STREAM) as client:
                client.settimeout(5)
                client.connect((address, port))
                conn, _ = server.accept()
                conn.close()
    finally:
        server.close()


def test_loopback_hosts():
    assert is_loopback("127.0.0.1") and is_loopback("localhost") and is_loopback("::1")
    assert not is_loopback("::") and not is_loopback("0.0.0.0")
