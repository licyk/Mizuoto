"""Training: experiments, their stages, metrics and exports."""

from fastapi import APIRouter

from mizuoto.api.deps import ServicesDep
from mizuoto.api.errors import ERROR_RESPONSES
from mizuoto.core.jobs.models import Job
from mizuoto.core.training.models import Experiment, ExperimentInput, ExperimentPatch, MethodInfo, MetricPoint, RunRequest

router = APIRouter(prefix="/v1/train", tags=["train"], responses=ERROR_RESPONSES)


@router.get("/methods", operation_id="list_training_methods")
def methods(services: ServicesDep, model: str) -> list[MethodInfo]:
    """The ways a model can be fine-tuned, and why one cannot."""
    return services.training.methods(model)


@router.get("/experiments", operation_id="list_experiments")
def list_experiments(services: ServicesDep) -> list[Experiment]:
    return services.training.list_experiments()


@router.post("/experiments", operation_id="create_experiment")
def create(services: ServicesDep, body: ExperimentInput) -> Experiment:
    return services.training.create(body)


@router.get("/experiments/{engine}/{name}", operation_id="get_experiment")
def get(services: ServicesDep, engine: str, name: str) -> Experiment:
    return services.training.get(f"{engine}/{name}")


@router.patch("/experiments/{engine}/{name}", operation_id="update_experiment")
def update(services: ServicesDep, engine: str, name: str, body: ExperimentPatch) -> Experiment:
    return services.training.update(f"{engine}/{name}", body)


@router.delete("/experiments/{engine}/{name}", operation_id="delete_experiment")
def delete(services: ServicesDep, engine: str, name: str) -> None:
    services.training.delete(f"{engine}/{name}")


@router.post("/experiments/{engine}/{name}/run", operation_id="run_experiment")
def run(services: ServicesDep, engine: str, name: str, body: RunRequest) -> Job:
    """Run stages as a ``train`` job: the given ones, or every stage from the first not done."""
    return services.training.run(f"{engine}/{name}", body, origin="api")


@router.get("/experiments/{engine}/{name}/metrics", operation_id="get_experiment_metrics")
def metrics(services: ServicesDep, engine: str, name: str, stage: str | None = None) -> list[MetricPoint]:
    return services.training.metrics(f"{engine}/{name}", stage)
