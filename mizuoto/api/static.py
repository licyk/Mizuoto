"""Serve the built web UI. Registered last, after the API and the socket."""

import logging
from pathlib import Path

from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

logger = logging.getLogger(__name__)


def web_dist_dir() -> Path:
    """Locate ``dist/`` through the package, so a checkout and an installed wheel both work."""
    import mizuoto.webui as web

    return Path(next(iter(web.__path__))) / "dist"


class SPAStaticFiles(StaticFiles):
    """Hashed files under ``assets/`` are cached for a year; ``index.html`` is never cached.

    Unknown paths fall back to ``index.html`` only for requests that accept HTML, so a mistyped
    asset or API path still gets a 404.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            response = await super().get_response(path, scope)
        except HTTPException as e:
            if e.status_code != 404 or not self._accepts_html(scope):
                raise
            response = await super().get_response("index.html", scope)
        else:
            if response.status_code == 404 and self._accepts_html(scope):
                response = await super().get_response("index.html", scope)
        if path.startswith("assets/") and response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-cache"
        return response

    @staticmethod
    def _accepts_html(scope: Scope) -> bool:
        for key, value in scope.get("headers", []):
            if key == b"accept":
                return b"text/html" in value
        return False
