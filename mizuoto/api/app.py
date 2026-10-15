"""``create_app(services)``: the FastAPI app with routers, the socket, and the web UI."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from mizuoto.api.errors import install_error_handlers
from mizuoto.api.openapi import MizuotoAPI
from mizuoto.api.routers import app_info, assets, compute, datasets, engines, jobs, models, openai, outputs, presets, projects, remote, runtime, settings, train, tts, voices
from mizuoto.api.routers import audio as audio_router
from mizuoto.api.security import SecurityMiddleware
from mizuoto.api.sockets import SocketBridge
from mizuoto.api.static import SPAStaticFiles, web_dist_dir
from mizuoto.core.context import Services
from mizuoto.version import VERSION

logger = logging.getLogger(__name__)

ROUTERS = (app_info, settings, compute, engines, runtime, models, assets, voices, audio_router, tts, outputs, presets, datasets, train, projects, remote, jobs)


def normalize_prefix(prefix: str | None) -> str:
    """``/voice/`` and ``voice`` both become ``/voice``; nothing becomes ``""``."""
    cleaned = (prefix or "").strip().strip("/")
    return f"/{cleaned}" if cleaned else ""


def create_app(
    services: Services,
    bound_host: str | None = None,
    bound_port: int | None = None,
    extra_hosts: set[str] | None = None,
    serve_ui: bool = True,
    api_prefix: str | None = None,
) -> FastAPI:
    """Build the application. The lifespan starts the socket bridge; the caller owns the services.

    ``api_prefix`` moves everything served here — the API, the socket and the web UI — under one
    path, so a host application can mount it beside its own routes. The web UI needs no change:
    it derives its base URL from the URL its own script was loaded from. A host that mounts the
    app enters its ``router.lifespan_context`` on the serving event loop and closes the services.

    ``extra_hosts`` are ``Host`` names accepted besides the loopback names and ``bound_host``, such
    as a tunnel's public name. The set is kept, not copied: names added to it later are accepted
    from then on.
    """
    prefix = normalize_prefix(api_prefix)
    socket_bridge = SocketBridge(services.events, lambda: services.settings.settings.server.access_token)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        await socket_bridge.start()
        try:
            yield
        finally:
            await socket_bridge.stop()

    app = MizuotoAPI(
        title="Mizuoto", version=VERSION, lifespan=lifespan, docs_url=f"{prefix}/docs", redoc_url=None, openapi_url=f"{prefix}/openapi.json", separate_input_output_schemas=False
    )
    app.state.services = services
    app.state.bound_port = bound_port
    app.state.api_prefix = prefix
    install_error_handlers(app)

    for module in ROUTERS:
        app.include_router(module.router, prefix=f"{prefix}/api")
    app.include_router(openai.router, prefix=prefix)
    app.mount(f"{prefix}/ws", socket_bridge.asgi_app(), name="socket")

    origins = services.settings.settings.server.allowed_origins
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"], allow_credentials=True)
    app.add_middleware(GZipMiddleware, minimum_size=4096)
    app.add_middleware(
        SecurityMiddleware,
        bound_host=lambda: bound_host or services.settings.settings.server.host,
        allowed_origins=lambda: services.settings.settings.server.allowed_origins,
        access_token=lambda: services.settings.settings.server.access_token,
        extra_hosts=extra_hosts,
        prefix=prefix,
    )

    # Registered last, so it only receives paths nothing else matched.
    dist = web_dist_dir()
    if serve_ui and (dist / "index.html").is_file():
        app.mount(f"{prefix}/", SPAStaticFiles(directory=dist, html=True), name="ui")
    elif serve_ui:
        logger.warning("No web UI build found at %s; serving the API only", dist)
    return app
