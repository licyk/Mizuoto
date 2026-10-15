"""Remote hosts: other machines running ``mizuoto agent`` whose engines this server uses."""

from fastapi import APIRouter

from mizuoto.api.deps import ServicesDep
from mizuoto.api.errors import ERROR_RESPONSES
from mizuoto.core.record import Record
from mizuoto.core.remote.models import RemoteHostStatus

router = APIRouter(prefix="/v1/remote", tags=["remote"], responses=ERROR_RESPONSES)


class RemoteHostInput(Record):
    address: str
    token: str | None = None
    """None keeps the saved token."""
    tls: bool = False
    ca_file: str = ""


class EngineHost(Record):
    host: str | None = None
    """None: this machine."""
    device: str = "auto"


class EngineHostResult(Record):
    device: str


@router.get("/hosts", operation_id="list_remote_hosts")
def list_hosts(services: ServicesDep, refresh: bool = False) -> list[RemoteHostStatus]:
    return services.remote.status(refresh=refresh)


@router.put("/hosts/{name}", operation_id="put_remote_host")
def put_host(services: ServicesDep, name: str, body: RemoteHostInput) -> RemoteHostStatus:
    return services.remote.add_host(name, body.address, body.token, body.tls, body.ca_file)


@router.delete("/hosts/{name}", operation_id="delete_remote_host")
def delete_host(services: ServicesDep, name: str) -> list[str]:
    """Remove a host; returns the engines that run on this machine again."""
    return services.remote.remove_host(name)


@router.put("/engines/{engine_id}", operation_id="set_engine_host")
def set_engine_host(services: ServicesDep, engine_id: str, body: EngineHost) -> EngineHostResult:
    return EngineHostResult(device=services.remote.use(engine_id, body.host, body.device))
