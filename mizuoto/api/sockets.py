"""Bridge the EventBus to socket.io. Traffic is server to client only."""

import asyncio
import contextlib
import hmac
import logging
from collections.abc import Callable
from typing import Any

import socketio
from starlette.types import ASGIApp, Receive, Scope, Send

from mizuoto.api.paths import route_path
from mizuoto.api.security import request_token
from mizuoto.core.events import EventBase, EventBus

logger = logging.getLogger(__name__)


class SocketBridge:
    def __init__(self, events: EventBus, access_token: Callable[[], str | None]) -> None:
        self.events = events
        self.access_token = access_token
        # Origin checks happen in SecurityMiddleware, in front of this app.
        self.sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*", logger=False, engineio_logger=False)
        self._queue: asyncio.Queue[EventBase] | None = None
        self._task: asyncio.Task[None] | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self.sio.on("connect", self._on_connect)

    def asgi_app(self) -> ASGIApp:
        """Normalize Starlette's mount scope before engineio matches its endpoint."""
        app = socketio.ASGIApp(self.sio, socketio_path="/socket.io")

        async def mounted(scope: Scope, receive: Receive, send: Send) -> None:
            await app({**scope, "path": route_path(scope)}, receive, send)

        return mounted

    async def _on_connect(self, sid: str, environ: dict[str, Any], auth: dict[str, Any] | None = None) -> bool:
        token = self.access_token()
        if not token:
            return True
        given = (auth or {}).get("token") if isinstance(auth, dict) else None
        if not given:
            scope = environ.get("asgi.scope", {})
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
            given = request_token(scope, headers)
        return bool(given) and hmac.compare_digest(str(given), token)

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        queue = self._queue

        def handler(event: EventBase) -> None:
            # Called from worker threads: cross into the loop thread-safely.
            loop.call_soon_threadsafe(queue.put_nowait, event)

        self._unsubscribe = self.events.subscribe(handler)
        self._task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            try:
                await self.sio.emit(event.__event_name__, event.model_dump(mode="json"))
            except Exception:
                logger.exception("Failed to emit %s", event.__event_name__)

    async def stop(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
