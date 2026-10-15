"""runtime: the models loaded in the running server (or in this command), and unloading them."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Annotated, Any

import typer

from mizuoto.cli.output import console, human_size, open_services, print_json, print_table

if TYPE_CHECKING:
    from mizuoto.core.context import Services
    from mizuoto.core.inference.models import LoadedModelInfo, RuntimeStatus


def _server(services: Services, method: str, path: str, body: dict[str, Any] | None = None) -> RuntimeStatus | None:
    """Ask the running server (``server.json`` in the data folder), whose process holds the models; None without one."""
    import httpx

    from mizuoto.core.inference.models import RuntimeStatus
    from mizuoto.core.net.runtime_file import read_runtime_file

    found = read_runtime_file(services.settings.data_dir)
    if not found or not found.get("url"):
        return None
    token = services.settings.settings.server.access_token
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = httpx.request(method, str(found["url"]).rstrip("/") + "/api/v1/runtime" + path, json=body, headers=headers, timeout=30.0)
    except httpx.HTTPError:
        return None
    if response.status_code == 404 and body is not None:
        from mizuoto.core.errors import NotFoundError

        raise NotFoundError(str(response.json().get("message") or "Not loaded"))
    response.raise_for_status()
    return RuntimeStatus.model_validate(response.json())


def runtime(
    model: Annotated[str | None, typer.Argument(help="With --unload: the model to unload (default: every model not in use)")] = None,
    unload: Annotated[bool, typer.Option("--unload", help="Unload a model, or every model not in use")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """List the loaded models (device, memory, attention backend, last use) and unload them.

    The models live in the server's process: with a server running (mizuoto webui) this asks it; otherwise this
    command's own process has nothing loaded.
    """
    if model is not None and not unload:
        raise typer.BadParameter("Name a model only with --unload")
    with open_services() as services:
        if unload:
            status = _server(services, "POST", "/unload", {"model": model}) or services.runtime.unload(model)
        else:
            status = _server(services, "GET", "") or services.runtime.status()
        if json_output:
            print_json(status)
            return
        if not status.models:
            console.print("No model is loaded" + ("" if status.torch else " (torch is not installed: mizuoto doctor)"), highlight=False)
            return
        now = time.time()
        print_table(
            None,
            ["Model", "Device", "Memory", "On device", "Attention", "Last used", "State"],
            [
                (
                    m.model_id,
                    m.device,
                    human_size(m.bytes) if m.bytes is not None else "-",
                    _on_device(m),
                    m.attention or "-",
                    f"{int((now - m.last_used) // 60)} min ago" if m.last_used else "-",
                    "busy" if m.busy else "idle",
                )
                for m in status.models
            ],
        )


def _on_device(m: LoadedModelInfo) -> str:
    """How much of a model is on its device: all of it, part (the rest streams from RAM), or none (moved to RAM)."""
    if m.loaded_bytes is None or m.bytes is None or m.device == "cpu":
        return "-"
    if m.loaded_bytes == 0:
        return "none (in RAM)"
    if m.streamed_bytes:
        return f"{human_size(m.loaded_bytes)} (rest streams)"
    return "all"
