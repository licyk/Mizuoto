"""Mizuoto: one platform for open TTS engines, with one web UI and one command line.

The application logic lives in :mod:`mizuoto.core` and can be used without the server::

    from mizuoto.core.context import open_services

    with open_services() as services:
        print(services.settings.view())

To embed the web UI and API in another application::

    from mizuoto import MizuotoServer

    server = MizuotoServer(data_dir="./mizuoto-data", port=0)
    print(server.start())   # http://127.0.0.1:54123
"""

from typing import TYPE_CHECKING, Any

from mizuoto.version import VERSION

if TYPE_CHECKING:
    from mizuoto.embed import MizuotoServer, serve

__version__ = VERSION
__all__ = ["VERSION", "MizuotoServer", "__version__", "serve"]

_LAZY = {"MizuotoServer", "serve"}


def __getattr__(name: str) -> Any:
    """Import the server lazily, so ``mizuoto version`` does not pay for FastAPI."""
    if name in _LAZY:
        from mizuoto import embed

        return getattr(embed, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
