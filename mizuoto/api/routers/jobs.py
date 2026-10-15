"""Jobs: every long operation."""

from fastapi import APIRouter, status

from mizuoto.api.deps import ServicesDep
from mizuoto.api.errors import ERROR_RESPONSES
from mizuoto.core.jobs.models import Job, JobLog, JobPage
from mizuoto.core.record import Record

router = APIRouter(prefix="/v1/jobs", tags=["jobs"], responses=ERROR_RESPONSES)


class ClearedJobs(Record):
    ids: list[str]


@router.get("", operation_id="list_jobs")
def list_jobs(services: ServicesDep, state: str | None = None, kind: str | None = None, cursor: str | None = None, limit: int = 50) -> JobPage:
    """``state`` is a job state, ``active`` or ``finished``."""
    return services.jobs.list_jobs(state, kind, cursor, max(1, min(limit, 200)))


@router.delete("", operation_id="clear_jobs")
def clear_jobs(services: ServicesDep, finished: bool = True) -> ClearedJobs:
    """Remove finished jobs and their logs. Their outputs stay."""
    return ClearedJobs(ids=services.jobs.clear_finished() if finished else [])


@router.get("/{job_id}", operation_id="get_job")
def get_job(services: ServicesDep, job_id: str) -> Job:
    return services.jobs.get(job_id)


@router.get("/{job_id}/log", operation_id="get_job_log")
def get_log(services: ServicesDep, job_id: str, offset: int = 0) -> JobLog:
    return services.jobs.read_log(job_id, max(0, offset))


@router.post("/{job_id}/cancel", operation_id="cancel_job")
def cancel(services: ServicesDep, job_id: str) -> Job:
    return services.jobs.cancel(job_id)


@router.post("/{job_id}/retry", operation_id="retry_job")
def retry(services: ServicesDep, job_id: str) -> Job:
    return services.jobs.retry(job_id)


@router.post("/{job_id}/run-anyway", operation_id="run_job_anyway")
def run_anyway(services: ServicesDep, job_id: str) -> Job:
    """Start a job that waits for a GPU slot now (you know the memory suffices)."""
    return services.jobs.run_anyway(job_id)


@router.delete("/{job_id}", operation_id="delete_job", status_code=status.HTTP_204_NO_CONTENT)
def delete(services: ServicesDep, job_id: str) -> None:
    services.jobs.delete(job_id)
