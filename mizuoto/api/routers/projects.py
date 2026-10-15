"""Projects: long-form scripts read by a cast of voices."""

from typing import Literal

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from mizuoto.api.deps import ServicesDep
from mizuoto.api.errors import ERROR_RESPONSES
from mizuoto.core.jobs.models import Job
from mizuoto.core.projects.models import ExportRequest, Project, ProjectInput, ProjectSummary, RenderRequest, ScriptImport
from mizuoto.core.record import Record
from mizuoto.core.tts.models import SpeakRequest

router = APIRouter(prefix="/v1/projects", tags=["projects"], responses=ERROR_RESPONSES)


class TakeChoice(Record):
    output_id: str


@router.get("", operation_id="list_projects")
def list_projects(services: ServicesDep) -> list[ProjectSummary]:
    return services.projects.list_projects()


@router.post("", operation_id="create_project")
def create(services: ServicesDep, body: ProjectInput) -> Project:
    return services.projects.create(body)


@router.get("/{project_id}", operation_id="get_project")
def get(services: ServicesDep, project_id: str) -> Project:
    return services.projects.get(project_id)


@router.put("/{project_id}", operation_id="update_project")
def update(services: ServicesDep, project_id: str, body: ProjectInput) -> Project:
    """Replace the settings, cast and script; lines keep their takes by id."""
    return services.projects.update(project_id, body)


@router.delete("/{project_id}", operation_id="delete_project", status_code=204)
def delete(services: ServicesDep, project_id: str) -> None:
    services.projects.delete(project_id)


@router.get("/{project_id}/script", operation_id="get_project_script", response_class=PlainTextResponse)
def script(services: ServicesDep, project_id: str) -> PlainTextResponse:
    return PlainTextResponse(services.projects.script(project_id))


@router.post("/{project_id}/script", operation_id="import_project_script")
def import_script(services: ServicesDep, project_id: str, body: ScriptImport) -> Project:
    return services.projects.import_script(project_id, body)


@router.post("/{project_id}/render", operation_id="render_project")
def render(services: ServicesDep, project_id: str, body: RenderRequest) -> Job:
    return services.projects.render(project_id, body, origin="api")


@router.post("/{project_id}/export", operation_id="export_project")
def export(services: ServicesDep, project_id: str, body: ExportRequest) -> Job:
    return services.projects.export(project_id, body, origin="api")


@router.put("/{project_id}/lines/{line_id}/take", operation_id="use_project_take")
def use_take(services: ServicesDep, project_id: str, line_id: str, body: TakeChoice) -> Project:
    return services.projects.use_take(project_id, line_id, body.output_id)


@router.get("/{project_id}/lines/{line_id}/request", operation_id="get_project_line_request")
def line_request(services: ServicesDep, project_id: str, line_id: str) -> SpeakRequest:
    """The synthesis request of a line, to try it in Speak."""
    return services.projects.request_for(project_id, line_id)


@router.get("/exports/{output_id}/subtitles", operation_id="get_project_subtitles", response_class=PlainTextResponse)
def subtitles(services: ServicesDep, output_id: str, format: Literal["srt", "vtt"] = "srt") -> PlainTextResponse:
    """An exported project's lines as subtitles."""
    media = "text/vtt" if format == "vtt" else "application/x-subrip"
    return PlainTextResponse(services.projects.subtitles(output_id, format), media_type=media)
