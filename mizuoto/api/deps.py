"""FastAPI dependencies. Services come from ``app.state`` rather than a global locator."""

from typing import Annotated

from fastapi import Depends, Request

from mizuoto.core import codes
from mizuoto.core.context import Services
from mizuoto.core.errors import MizuotoError
from mizuoto.core.net.ports import is_loopback


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


class NotLocalError(MizuotoError):
    code = codes.NOT_LOCAL
    http_status = 403
    exit_code = 4


def is_local(request: Request) -> bool:
    """A browser on the server's own machine: a loopback client that addressed a loopback name.

    A tunnel or a proxy on the same machine connects from loopback too, but forwards its public
    ``Host``; its visitors are elsewhere.
    """
    client = request.client.host if request.client else ""
    if client == "testclient":
        return True
    return bool(client) and is_loopback(client) and is_loopback(request.url.hostname or "")


def require_local(request: Request) -> None:
    """Revealing files and reading server paths work only for a browser on the server's own machine."""
    if not is_local(request):
        raise NotLocalError("This works only in a browser on the server's own machine")


LocalDep = Depends(require_local)
