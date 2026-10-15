"""agent (serve this machine's engines) and remote list | add | remove | use (another machine's)."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Annotated, Any

import typer

from mizuoto.cli.output import console, err_console, open_services, print_json, print_table


def agent(
    host: Annotated[str | None, typer.Option(help="Address to listen on (default from settings: 127.0.0.1; 0.0.0.0 for every interface)")] = None,
    port: Annotated[int | None, typer.Option(min=0, max=65535, help="Port (default from settings: 7870)")] = None,
    token: Annotated[str | None, typer.Option(help="The token hosts must give (default from settings; made and saved when there is none)")] = None,
    ssl_certfile: Annotated[Path | None, typer.Option(help="PEM certificate: serve TLS (default from settings: remote.agent.ssl_certfile)", exists=True, dir_okay=False)] = None,
    ssl_keyfile: Annotated[Path | None, typer.Option(help="The certificate's private key", exists=True, dir_okay=False)] = None,
) -> None:
    """Serve this machine's engines to another Mizuoto (its engines then run here)."""
    from mizuoto.core.context import build_services
    from mizuoto.core.remote.agent import AgentServer
    from mizuoto.core.remote.wire import server_context

    services = build_services(start_background=True)
    try:
        a = services.settings.settings.remote.agent
        if token is None and not a.token:
            token = secrets.token_urlsafe(24)
            services.settings.update({"remote": {"agent": {"token": token}}})
            err_console.print(f"Made a token and saved it (remote.agent.token): [bold]{token}[/bold]", highlight=False)
        certfile = str(ssl_certfile) if ssl_certfile else a.ssl_certfile
        keyfile = str(ssl_keyfile) if ssl_keyfile else a.ssl_keyfile
        server = AgentServer(services, host or a.host, a.port if port is None else port, token or a.token or "", server_context(certfile, keyfile) if certfile else None)
        server.start()
        bound_host, bound_port = server.address
        engines = [e.id for e in server.info().engines if e.ready]
        console.print(f"Mizuoto agent on {bound_host}:{bound_port}{' (TLS)' if certfile else ''}: {', '.join(engines) or 'no engine installed yet'}", highlight=False)
        if not certfile and bound_host not in ("127.0.0.1", "::1", "localhost"):
            err_console.print("[yellow]warning:[/yellow] without TLS, requests and audio cross the network unencrypted; use a private network or a tunnel", highlight=False)
        console.print(f"On the host: mizuoto remote add <name> <this machine>:{bound_port} --token …", highlight=False)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    finally:
        services.close()


def _status_rows(statuses: list[Any]) -> list[tuple[Any, ...]]:
    rows = []
    for s in statuses:
        info = s.info
        engines = ", ".join(e.id for e in info.engines if e.ready) if info else "-"
        devices = ", ".join(d.id for d in info.devices) if info else "-"
        rows.append((s.name, s.address, s.state, info.version if info else (s.error or "-"), devices, engines, ", ".join(s.engines) or "-"))
    return rows


def remote_list(json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """List the remote hosts and what each has."""
    with open_services() as services:
        statuses = services.remote.status(refresh=True)
        if json_output:
            print_json(statuses)
            return
        print_table(None, ["Name", "Address", "State", "Version", "Devices", "Engines", "Used by"], _status_rows(statuses))


def remote_add(
    name: Annotated[str, typer.Argument(help="A short name; engines name it in their device: <name>/cuda:0")],
    address: Annotated[str, typer.Argument(help="host or host:port of the agent (port 7870 by default)")],
    token: Annotated[str | None, typer.Option("--token", help="The agent's token (shown when `mizuoto agent` made it; kept when changing a host)")] = None,
    tls: Annotated[bool, typer.Option("--tls", help="The agent serves TLS")] = False,
    ca_file: Annotated[Path | None, typer.Option("--ca-file", help="A CA bundle, or the agent's own certificate when self-signed", exists=True, dir_okay=False)] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Add (or change) a remote host and check that it answers."""
    with open_services() as services:
        status = services.remote.add_host(name, address, token, tls, str(ca_file) if ca_file else "")
        if json_output:
            print_json(status)
            return
        if status.state == "online" and status.info is not None:
            ready = [e.id for e in status.info.engines if e.ready]
            console.print(f"{name}: {status.info.name}, Mizuoto {status.info.version}, {', '.join(d.id for d in status.info.devices)}", highlight=False)
            console.print(f"Engines there: {', '.join(ready) or 'none installed'}. Use one with: mizuoto remote use <engine> {name} --device cuda:0", highlight=False)
        else:
            err_console.print(f"[yellow]Saved, but {name} is {status.state}:[/yellow] {status.error}", highlight=False)


def remote_remove(name: Annotated[str, typer.Argument(help="The host's name")]) -> None:
    """Remove a remote host; engines that used it go back to this machine."""
    with open_services() as services:
        engines = services.remote.remove_host(name)
        console.print(f"Removed {name}" + (f"; {', '.join(engines)} run here again" if engines else ""), highlight=False)


def remote_use(
    engine: Annotated[str, typer.Argument(help="An engine id (mizuoto engine list)")],
    name: Annotated[str | None, typer.Argument(help="The remote host (leave out with --local)")] = None,
    device: Annotated[str, typer.Option("--device", "-d", help="The device on that host: auto, cuda:0, cpu…")] = "auto",
    local: Annotated[bool, typer.Option("--local", help="Run the engine on this machine again")] = False,
) -> None:
    """Run an engine on a remote host (its packages and models are that host's)."""
    if name is None and not local:
        raise typer.BadParameter("Name a remote host, or give --local")
    with open_services() as services:
        value = services.remote.use(engine, None if local else name, device)
        if local:
            console.print(f"{engine} runs on this machine", highlight=False)
            return
        status = services.engines.status(engine)
        console.print(f"{engine} runs on {value}" + ("" if status.available else f" [yellow](not available there: on {name}, mizuoto doctor {engine})[/yellow]"), highlight=False)
