"""Request paths, the same on every Starlette version."""

from starlette import _utils
from starlette.types import Scope


def route_path(scope: Scope) -> str:
    """Older Starlette strips mount prefixes; newer versions leave that to the router."""
    resolve = getattr(_utils, "get_route_path", None)
    return resolve(scope) if resolve is not None else scope.get("path", "")
