"""Run Mizuoto inside another application.

    from mizuoto import MizuotoServer

    server = MizuotoServer(
        data_dir="./mizuoto-data",                   # database, models, outputs, caches
        config_dir="./my-app/config/Mizuoto",        # settings.toml goes here instead of data_dir
        settings={"paths": {"models_dir": "/srv/tts/models"}},  # pinned: the UI cannot change them
        port=0,                                       # 0: any free port
        api_prefix="/voice",                          # keeps our routes out of yours
    )
    url = server.start()   # returns as soon as it is listening: http://127.0.0.1:54123/voice
    ...
    server.stop()

``server.run()`` serves in the foreground instead. The object is also a context manager, and
``server.services`` exposes the voice library, the jobs and the settings for direct use.

Behind a tunnel or a proxy, ``server.allow_host("abc.example.com")`` accepts its public name, even
once started. Anyone with the address can then reach the server; ``access_token`` asks them for a
token, or put the server behind protection of your own.
"""

import logging
import threading
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

from mizuoto.core.context import Services, build_services
from mizuoto.core.net.ports import bind_first_free_port, is_loopback

logger = logging.getLogger(__name__)


def _host_name(host: str) -> str:
    """``abc.example.com`` from itself, ``abc.example.com:443`` or ``https://abc.example.com/path``."""
    host = host.strip()
    return (urlsplit(host if "//" in host else f"//{host}").hostname or "").lower()


START_TIMEOUT = 30.0


class MizuotoServer:
    """The web UI and API, embedded in another program."""

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        config_dir: str | Path | None = None,
        settings_path: str | Path | None = None,
        host: str | None = None,
        port: int | None = None,
        strict_port: bool = False,
        api_prefix: str | None = None,
        open_browser: bool = False,
        access_token: str | None = None,
        extra_hosts: Iterable[str] | None = None,
        settings: dict[str, Any] | None = None,
        log_level: str = "warning",
    ) -> None:
        """
        ``data_dir`` holds the database, the models, experiments and outputs, and the caches; by
        default it also holds ``settings.toml``. ``config_dir`` puts ``settings.toml`` in a folder
        of your own — your application's configuration folder, for instance — and
        ``settings_path`` names the file itself. The data stays in ``data_dir`` either way.

        ``port`` is ``0`` for any free port, a number to ask for that one (moving up when it is
        taken, unless ``strict_port``), or ``None`` for the port in the settings.

        ``api_prefix`` moves the API, the socket and the web UI under one path, so they cannot
        collide with the host application's own routes.

        ``extra_hosts`` are public names the server answers to besides its own address: a tunnel's
        or a reverse proxy's. Requests through them are never treated as coming from the server's
        own machine.

        ``access_token``, when given, must accompany every API and socket request (the web UI asks
        for it once). It is optional, a non-loopback ``host`` and ``extra_hosts`` included.

        ``settings`` pins any setting, in the same shape as the settings file, for example
        ``{"compute": {"default_device": "cuda:0"}, "paths": {"models_dir": "/srv/voices"}}``. Pinned
        values are never written to the file and cannot be changed through the UI or the API.
        """
        self.data_dir = Path(data_dir).expanduser() if data_dir else None
        self.config_dir = Path(config_dir).expanduser() if config_dir else None
        self.settings_path = Path(settings_path).expanduser() if settings_path else None
        self.api_prefix = api_prefix or ""
        self.open_browser = open_browser
        self.strict_port = strict_port
        self._requested_port = port
        self._log_level = log_level
        # Shared with the security middleware, so allow_host() works while serving.
        self._extra_hosts: set[str] = {_host_name(h) for h in extra_hosts or ()}

        overrides: dict[str, Any] = {k: dict(v) if isinstance(v, dict) else v for k, v in (settings or {}).items()}
        server_overrides: dict[str, Any] = dict(overrides.get("server") or {})
        if host is not None:
            server_overrides["host"] = host
        if access_token is not None:
            server_overrides["access_token"] = access_token
        server_overrides["open_browser"] = open_browser
        overrides["server"] = server_overrides
        self._overrides = overrides

        self.services: Services | None = None
        self._socket: Any = None
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._url: str | None = None

    # -- lifecycle ----------------------------------------------------------

    def _build(self) -> Services:
        return build_services(
            data_dir=self.data_dir,
            config_dir=self.config_dir,
            settings_path=self.settings_path,
            settings_overrides=self._overrides,
            start_background=True,
        )

    def _bind(self, services: Services) -> Any:
        server_settings = services.settings.settings.server
        host = server_settings.host
        port = server_settings.port if self._requested_port is None else self._requested_port
        if not is_loopback(host):
            logger.warning("Binding to %s makes the server reachable from other machines.", host)
        return bind_first_free_port(host, port, strict=self.strict_port or port == 0)

    def start(self, timeout: float = START_TIMEOUT) -> str:
        """Start serving in a background thread and return the URL of the web UI."""
        import uvicorn

        from mizuoto.api.app import create_app
        from mizuoto.core.net.runtime_file import write_runtime_file

        if self._thread is not None:
            return self.url
        services = self._build()
        self.services = services
        try:
            self._socket = self._bind(services)
            host = services.settings.settings.server.host
            port = self._socket.getsockname()[1]
            self._url = self._public_url(host, port)
            app = create_app(services, bound_host=host, bound_port=port, extra_hosts=self._extra_hosts, api_prefix=self.api_prefix)
            tls = services.settings.settings.server.ssl()
            self._server = uvicorn.Server(
                uvicorn.Config(app=app, log_level=self._log_level, lifespan="on", ssl_certfile=tls.get("ssl_certfile"), ssl_keyfile=tls.get("ssl_keyfile"))
            )
            sock = self._socket
            self._thread = threading.Thread(target=lambda: self._server.run(sockets=[sock]), name="Mizuoto", daemon=True)
            self._thread.start()
            self._wait_until_started(timeout)
            # Tells a command line opened on the same data directory that a server owns its jobs.
            write_runtime_file(services.settings.data_dir, host, port, self._url)
        except BaseException:
            self.stop()
            raise
        if self.open_browser:
            import webbrowser

            threading.Timer(0.5, lambda: webbrowser.open(self._url or "")).start()
        logger.info("Mizuoto is running on %s", self._url)
        return self._url

    def _wait_until_started(self, timeout: float) -> None:
        pause = threading.Event()
        waited = 0.0
        while waited < timeout:
            if getattr(self._server, "started", False):
                return
            if self._thread is not None and not self._thread.is_alive():
                raise RuntimeError("The server stopped while starting; see the log for the reason.")
            pause.wait(0.05)
            waited += 0.05
        raise TimeoutError(f"The server did not start within {timeout} seconds")

    def run(self) -> None:
        """Serve in the foreground until interrupted."""
        self.start()
        try:
            while self._thread is not None and self._thread.is_alive():
                self._thread.join(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self, timeout: float = 10.0) -> None:
        """Stop serving and release everything, the live worker included. Safe to call more than once."""
        from mizuoto.core.net.runtime_file import remove_runtime_file

        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        self._server = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self.services is not None:
            remove_runtime_file(self.services.settings.data_dir)
            self.services.close()
            self.services = None
        self._url = None

    def allow_host(self, host: str) -> None:
        """Answer to ``host`` too: a tunnel's or a proxy's public name (a URL is accepted). Works before
        and after ``start()``."""
        name = _host_name(host)
        if not name:
            raise ValueError(f"Not a host name: {host!r}")
        self._extra_hosts.add(name)

    # -- information --------------------------------------------------------

    @property
    def url(self) -> str:
        """Where the web UI is, once started."""
        if not self._url:
            raise RuntimeError("The server is not running; call start() first.")
        return self._url

    @property
    def port(self) -> int:
        return int(self.url.rsplit(":", 1)[1].split("/")[0])

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _public_url(self, host: str, port: int) -> str:
        from mizuoto.api.app import normalize_prefix

        shown = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
        if ":" in shown and not shown.startswith("["):
            shown = f"[{shown}]"
        scheme = "https" if self.services is not None and self.services.settings.settings.server.ssl_certfile else "http"
        return f"{scheme}://{shown}:{port}{normalize_prefix(self.api_prefix)}"

    # typing.Self needs Python 3.11; this package supports 3.10.
    def __enter__(self) -> "MizuotoServer":
        self.start()
        return self

    def __exit__(self, _type: type[BaseException] | None, _value: BaseException | None, _tb: TracebackType | None) -> None:
        self.stop()


def serve(block: bool = False, **options: Any) -> MizuotoServer:
    """Start a server with ``MizuotoServer`` options. With ``block`` it serves in the foreground.

    Returns the server, so a caller can read ``server.url`` and later ``server.stop()``.
    """
    server = MizuotoServer(**options)
    if block:
        server.run()
    else:
        server.start()
    return server


__all__ = ["MizuotoServer", "serve"]
