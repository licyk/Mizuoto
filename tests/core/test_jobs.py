import sys
import time
from pathlib import Path

import pytest

from mizuoto.core.errors import ValidationError
from mizuoto.core.jobs.service import INFERENCE, JobSpec
from mizuoto.core.process import WorkerTarget

JOB_WORKER = """
import time
from mizuoto.core.jobs.events import ProgressEvent, ResultEvent, ErrorEvent

def main(request, emitter):
    print("a stray print from a library")
    if request.get("sleep"):
        time.sleep(request["sleep"])
    if request.get("fail"):
        emitter.emit(ErrorEvent(code="model_missing", message="no weights", detail={"models": []}))
        raise SystemExit(1)
    emitter.emit(ProgressEvent(value=0.5, step="half", message="halfway"))
    emitter.emit(ResultEvent(data={"echo": request["value"]}))
"""


@pytest.fixture
def job_target(tmp_path: Path) -> WorkerTarget:
    (tmp_path / "fake_job_worker.py").write_text(JOB_WORKER)
    return WorkerTarget(Path(sys.executable), "mizuoto.training.stage", tmp_path, {"PYTHONPATH": str(tmp_path)})


def register(services, kind, run, resources=frozenset({"cpu"})):
    services.jobs.register(kind, lambda req: JobSpec(title=f"{kind} {req.get('n', '')}", run=lambda ctx: run(ctx, req), resources=resources))


def test_run_worker_reports_and_keeps_stray_output(services, job_target):
    register(services, "speak", lambda ctx, req: ctx.run_worker(job_target, req, args=["--entry", "fake_job_worker:main"]))
    job = services.jobs.run_now("speak", {"value": 42})
    assert job.state == "completed", job.error
    assert job.result == {"echo": 42}
    log = services.jobs.read_log(job.id)
    assert "a stray print from a library" in log.lines


def test_worker_error_becomes_the_domain_error(services, job_target):
    register(services, "speak", lambda ctx, req: ctx.run_worker(job_target, req, args=["--entry", "fake_job_worker:main"]))
    job = services.jobs.run_now("speak", {"value": 1, "fail": True})
    assert job.state == "failed" and job.error.code == "model_missing"
    assert job.can_retry


def test_silent_worker_times_out(services, job_target):
    register(services, "speak", lambda ctx, req: ctx.run_worker(job_target, req, args=["--entry", "fake_job_worker:main"], heartbeat_timeout=1.0))
    job = services.jobs.run_now("speak", {"value": 1, "sleep": 30})
    assert job.state == "failed" and job.error.code == "timeout"


def test_cancel_kills_a_running_worker(make_services, job_target):
    services = make_services(start_background=True)
    register(services, "speak", lambda ctx, req: ctx.run_worker(job_target, req, args=["--entry", "fake_job_worker:main"]))
    job = services.jobs.submit("speak", {"value": 1, "sleep": 30})
    deadline = time.monotonic() + 10
    while services.jobs.get(job.id).state != "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)
    services.jobs.cancel(job.id)
    while services.jobs.get(job.id).state not in ("cancelled", "failed") and time.monotonic() < deadline:
        time.sleep(0.05)
    assert services.jobs.get(job.id).state == "cancelled"


def test_named_resources_serialise_a_worker(make_services):
    services = make_services(start_background=True)
    running: list[str] = []
    overlap: list[bool] = []

    def run(ctx, req):
        running.append(req["n"])
        overlap.append(len(running) > 1)
        time.sleep(0.2)
        running.remove(req["n"])
        return {}

    register(services, "speak", run, resources=frozenset({"engine:dummy:cpu"}))
    jobs = [services.jobs.submit("speak", {"n": str(i)}) for i in range(3)]
    deadline = time.monotonic() + 10
    while any(services.jobs.get(j.id).state != "completed" for j in jobs) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert all(services.jobs.get(j.id).state == "completed" for j in jobs)
    assert not any(overlap)


def test_active_jobs_become_interrupted_after_a_restart(make_services, tmp_path):
    first = make_services()
    register(first, "speak", lambda ctx, req: {})
    job = first.jobs.submit("speak", {})  # queued; no scheduler runs in this process
    first.close()
    second = make_services()
    assert second.jobs.get(job.id).state == "interrupted"


def test_unknown_kind_is_refused(services):
    with pytest.raises(ValidationError):
        services.jobs.submit("teleport", {})


def test_a_new_stage_is_published_at_once(services):
    from mizuoto.core.events.models import JobUpdatedEvent

    seen = []
    services.events.subscribe(lambda e: seen.append(e.job.message) if isinstance(e, JobUpdatedEvent) else None)

    def run(ctx, req):
        for text in ("Loading", "Preparing", "Speaking", "Speaking", "Speaking"):
            ctx.message(text)
        return {}

    register(services, "speak", run)
    services.jobs.run_now("speak", {})
    assert [m for m in seen if m in ("Loading", "Preparing", "Speaking")][:3] == ["Loading", "Preparing", "Speaking"]


def test_another_live_process_keeps_its_jobs(make_services):
    import os

    first = make_services()
    register(first, "speak", lambda ctx, req: {})
    alive = first.jobs.submit("speak", {})
    dead = first.jobs.submit("speak", {})
    # As if a command line in another process (still running, or gone) had submitted them.
    first.db.execute("UPDATE jobs SET owner_pid = ? WHERE id = ?", (os.getppid(), alive.id))
    first.db.execute("UPDATE jobs SET owner_pid = ? WHERE id = ?", (2**22 + 12345, dead.id))
    first.close()
    second = make_services()
    assert second.jobs.get(alive.id).state == "queued"
    assert second.jobs.get(dead.id).state == "interrupted"


def test_an_exclusive_resource_takes_every_slot(make_services):
    """A fine-tune holds the whole GPU: with two GPU slots, nothing else runs beside it."""
    services = make_services(start_background=True, settings_overrides={"compute": {"gpu_jobs": 2}})
    running: list[str] = []
    overlap: list[tuple[str, ...]] = []

    def run(ctx, req):
        running.append(req["n"])
        overlap.append(tuple(sorted(running)))
        time.sleep(0.2)
        running.remove(req["n"])
        return {}

    services.jobs.register("train", lambda req: JobSpec(title="train", run=lambda ctx: run(ctx, req), resources=frozenset({"gpu:cuda:0"}), exclusive=frozenset({"gpu:cuda:0"})))
    register(services, "speak", run, resources=frozenset({"gpu:cuda:0"}))
    jobs = [services.jobs.submit("speak", {"n": "a"}), services.jobs.submit("speak", {"n": "b"}), services.jobs.submit("train", {"n": "t"})]
    jobs.append(services.jobs.submit("speak", {"n": "c"}))
    deadline = time.monotonic() + 10
    while any(services.jobs.get(j.id).state != "completed" for j in jobs) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert all(services.jobs.get(j.id).state == "completed" for j in jobs)
    assert ("a", "b") in overlap  # two speak jobs share the GPU
    assert not any("t" in group and len(group) > 1 for group in overlap)  # the fine-tune never does


@pytest.mark.parametrize("limit", [1, 2])
def test_synthesis_jobs_wait_for_the_inference_limit(make_services, limit):
    """compute.inference_jobs bounds synthesis jobs at once, even on different engines and on the CPU, where no
    other resource holds them back; the others wait in the queue and all run in the end."""
    services = make_services(start_background=True, settings_overrides={"compute": {"inference_jobs": limit}})
    running: list[str] = []
    most = [0]

    def run(ctx, req):
        running.append(req["n"])
        most[0] = max(most[0], len(running))
        time.sleep(0.15)
        running.remove(req["n"])
        return {}

    services.jobs.register("speak", lambda req: JobSpec(title=req["n"], run=lambda ctx: run(ctx, req), resources=frozenset({INFERENCE, f"engine:{req['n']}:cpu"})))
    jobs = [services.jobs.submit("speak", {"n": e}) for e in ("a", "b", "c")]
    assert services.jobs.capacity(INFERENCE) == limit
    done = [services.jobs.wait(j.id, 10) for j in jobs]
    assert all(j.state == "completed" for j in done)
    assert most[0] == limit
