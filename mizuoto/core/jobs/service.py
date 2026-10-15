"""The job service: queue, resources, runners and history.

Every long operation is a job. A kind registers a factory that turns a request (plain JSON) into a
``JobSpec``; the request is stored, so a job can be retried after a restart. The server runs jobs on
threads chosen by a scheduler; the command line runs one job in the foreground.

Resources are named, each with a capacity: ``cpu`` (half the cores), ``io`` (4), ``gpu:<device>``
(``compute.gpu_jobs``), ``inference`` (``compute.inference_jobs``: synthesis jobs at once, on any engine
and device) and ``engine:<engine>:<device>`` (1: a loaded engine serves one request at a time; a
training run holds it too).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mizuoto.core.clock import new_id, now_iso
from mizuoto.core.compute.service import ComputeService, MemoryHolder
from mizuoto.core.db import Database
from mizuoto.core.errors import CancelledError, ComputeError, ConflictError, MizuotoError, NotFoundError, OperationTimeoutError, ValidationError, WorkerError, error_from_code
from mizuoto.core.events import EventBus
from mizuoto.core.events.models import JobLogEvent, JobsClearedEvent, JobUpdatedEvent
from mizuoto.core.jobs.models import ACTIVE_STATES, FINISHED_STATES, ErrorInfo, Job, JobLog, JobPage, JobStep, Transfer
from mizuoto.core.process import WorkerTarget, describe_exit, kill_process_tree, start_target
from mizuoto.core.settings import SettingsService

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL = 0.25
# Download speed is measured over this many seconds.
TRANSFER_WINDOW = 5.0
LOG_EVENT_INTERVAL = 0.25
DB_PROGRESS_INTERVAL = 2.0
MAX_LOG_EVENT_LINES = 50
JOBS_HOLDER = "jobs"
INFERENCE = "inference"
"""The resource every synthesis job holds (``compute.inference_jobs`` of them run at once)."""
WORKER_HEARTBEAT_TIMEOUT = 120.0
"""A job worker silent for this long (no event, no line) is considered hung."""


class JobCancelled(Exception):
    """Raised inside a job when it was cancelled."""


@dataclass
class JobSpec:
    title: str
    run: Callable[[JobContext], dict[str, Any] | None]
    resources: frozenset[str] = frozenset({"cpu"})
    """Named resources the job holds while it runs (see the module docstring)."""
    steps: list[JobStep] = field(default_factory=list)
    priority: int = 0
    """Higher runs first; previews use 10."""
    group: str | None = None
    """Jobs with the same group can be run together by one of them (``JobContext.adopt``): speak requests
    that differ only in their text, for an engine that takes a batch."""
    payload: Any = None
    """What the adopting job needs of an adopted one (its prepared request)."""
    exclusive: frozenset[str] = frozenset()
    """Resources among ``resources`` the job takes whole: every slot (a fine-tune takes the GPU, whatever
    ``compute.gpu_jobs`` allows)."""


JobFactory = Callable[[dict[str, Any]], JobSpec]


@dataclass
class _Active:
    job: Job
    spec: JobSpec
    cancel: threading.Event = field(default_factory=threading.Event)
    run_anyway: bool = False
    process: Any = None
    last_progress_emit: float = 0.0
    last_db_write: float = 0.0
    pending_lines: list[str] = field(default_factory=list)
    pending_offset: int | None = None
    last_log_emit: float = 0.0
    transfer_samples: deque[tuple[float, int]] = field(default_factory=deque)
    adopted_by: str | None = None
    """The job running this one with its own (``JobContext.adopt``)."""
    taken: dict[str, int] = field(default_factory=dict)
    """Slots of each resource the running job holds."""


class JobContext:
    """What a running job uses to report and to check for cancellation."""

    def __init__(self, service: JobService, active: _Active) -> None:
        self._service = service
        self._active = active

    @property
    def job_id(self) -> str:
        return self._active.job.id

    @property
    def cancelled(self) -> bool:
        return self._active.cancel.is_set()

    def check_cancel(self) -> None:
        if self._active.cancel.is_set():
            raise JobCancelled()

    @property
    def cancel_event(self) -> threading.Event:
        """Set when the job is cancelled; pass it to anything that can stop early."""
        return self._active.cancel

    def message(self, text: str | None) -> None:
        """What is happening now, in words."""
        self._service._message(self._active, text)

    def progress(self, value: float | None, step: str | None = None) -> None:
        self._service._progress(self._active, value, step)

    def step(self, step_id: str, state: str, progress: float | None = None) -> None:
        self._service._step(self._active, step_id, state, progress)

    def transfer(self, done_bytes: int, total_bytes: int) -> None:
        """Bytes downloaded so far out of ``total_bytes``; the job gets a speed and an ETA from them."""
        self._service._transfer(self._active, done_bytes, total_bytes)

    def add_steps(self, steps: list[JobStep]) -> None:
        self._service._add_steps(self._active, steps)

    def log(self, line: str) -> None:
        self._service._log(self._active, line)

    def run_worker(
        self,
        target: WorkerTarget,
        request: dict[str, Any],
        args: list[str] | None = None,
        on_event: Callable[[Any], None] | None = None,
        heartbeat_timeout: float = WORKER_HEARTBEAT_TIMEOUT,
    ) -> dict[str, Any]:
        """Run ``<python> -m <module> <args…> <request.json>`` (a job worker, usually in an engine's
        environment); return its result event's data.

        Progress, step and log events are applied to the job unless ``on_event`` handles them (it
        sees every event first). An error event becomes the matching domain error. A worker that
        prints nothing for ``heartbeat_timeout`` seconds is killed.
        """
        return self._service._run_worker(self._active, target, request, list(args or []), on_event, heartbeat_timeout)

    @property
    def payload(self) -> Any:
        return self._active.spec.payload

    def adopt(self, limit: int) -> list[JobContext]:
        """Take up to ``limit`` queued jobs of this job's group, oldest first, to run with this one: they become
        running, and this job ends each with ``complete``, ``fail`` or ``release`` (back to the queue)."""
        return [JobContext(self._service, a) for a in self._service._adopt(self._active, limit)]

    def complete(self, result: dict[str, Any]) -> None:
        """End an adopted job (cancelled meanwhile, it ends cancelled)."""
        self._service._end_adopted(self._active, result=result)

    def fail(self, error: BaseException) -> None:
        self._service._end_adopted(self._active, error=error)

    def release(self) -> None:
        """Give an adopted job back to the queue, to run on its own."""
        self._service._release(self._active)


class JobService:
    def __init__(self, settings: SettingsService, db: Database, events: EventBus, compute: ComputeService) -> None:
        self._settings = settings
        self._db = db
        self._events = events
        self._compute = compute
        self._logs = settings.jobs_dir
        self._factories: dict[str, JobFactory] = {}
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._active: dict[str, _Active] = {}
        self._running: dict[str, int] = {}
        self._thread: threading.Thread | None = None
        # Freeing memory waits while a job that uses a device is queued or running.
        compute.add_holder(MemoryHolder(JOBS_HOLDER, busy=self._busy_reason))
        self._stop = False
        self._mark_interrupted()

    # -- registration --------------------------------------------------------

    def register(self, kind: str, factory: JobFactory) -> None:
        self._factories[kind] = factory

    def capacity(self, resource: str) -> int:
        if resource.startswith("gpu:"):
            return self._settings.settings.compute.gpu_jobs
        if resource == INFERENCE:
            return self._settings.settings.compute.inference_jobs
        if resource == "cpu":
            return max(1, (os.cpu_count() or 2) // 2)
        if resource == "io":
            return 4
        return 1

    def _busy_reason(self) -> str | None:
        with self._lock:
            n = sum(1 for a in self._active.values() if any(r.startswith(("gpu:", "engine:")) for r in a.spec.resources))
        return f"{n} job(s)" if n else None

    # -- submission ----------------------------------------------------------

    def submit(self, kind: str, request: dict[str, Any], origin: str | None = None) -> Job:
        """Queue a job; the scheduler runs it when its resources are free."""
        active = self._create(kind, request, origin)
        with self._cond:
            self._active[active.job.id] = active
            self._cond.notify_all()
        self._publish(active.job)
        self._compute.emit_usage(force=True)
        return active.job.model_copy(deep=True)

    def run_now(self, kind: str, request: dict[str, Any], origin: str | None = None) -> Job:
        """Run a job on the calling thread (the command line, Python callers). Ctrl+C cancels it."""
        active = self._create(kind, request, origin)
        with self._lock:
            self._active[active.job.id] = active
        self._publish(active.job)
        self._compute.emit_usage(force=True)
        try:
            self._execute(active)
        except KeyboardInterrupt:
            active.cancel.set()
            kill_process_tree(active.process)
            self._finish(active, "cancelled")
        return active.job.model_copy(deep=True)

    def _create(self, kind: str, request: dict[str, Any], origin: str | None) -> _Active:
        factory = self._factories.get(kind)
        if factory is None:
            raise ValidationError(f"Unknown job kind: {kind}")
        spec = factory(request)
        job = Job(
            id=new_id("j"),
            kind=kind,  # ty: ignore[invalid-argument-type]
            title=spec.title,
            state="queued",
            steps=[s.model_copy() for s in spec.steps],
            request=request,
            origin=origin,
            created_at=now_iso(),
            can_run_anyway=False,
        )
        self._db.execute(
            "INSERT INTO jobs(id, kind, title, state, request, progress, step, steps, origin, created_at, owner_pid) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job.id, kind, job.title, job.state, json.dumps(request), None, None, json.dumps([s.model_dump() for s in job.steps]), origin, job.created_at, os.getpid()),
        )
        return _Active(job=job, spec=spec)

    # -- scheduling ----------------------------------------------------------

    @property
    def scheduling(self) -> bool:
        """Whether the scheduler runs submitted jobs (the server); else only ``run_now`` runs anything."""
        return self._thread is not None

    def start(self) -> None:
        if self._thread is None:
            self._stop = False
            self._thread = threading.Thread(target=self._schedule_loop, name="jobs", daemon=True)
            self._thread.start()

    def close(self) -> None:
        with self._cond:
            self._stop = True
            actives = list(self._active.values())
            self._cond.notify_all()
        for a in actives:
            a.cancel.set()
            kill_process_tree(a.process)
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _schedule_loop(self) -> None:
        while True:
            with self._cond:
                if self._stop:
                    return
                self._schedule()
                self._cond.wait(timeout=1.0)

    def _schedule(self) -> None:
        pending = sorted((a for a in self._active.values() if a.job.state in ("queued", "waiting")), key=lambda a: (-a.spec.priority, a.job.created_at))
        for a in pending:
            full = None if a.run_anyway else next((r for r in sorted(a.spec.resources) if self._running.get(r, 0) >= (1 if r in a.spec.exclusive else self.capacity(r))), None)
            if full is not None:
                if a.job.state != "waiting" or a.job.waiting_for != full:
                    a.job.state = "waiting"
                    a.job.waiting_for = full
                    a.job.can_run_anyway = full.startswith("gpu:")
                    self._persist(a)
                    self._publish(a.job)
                continue
            for res in a.spec.resources:
                a.taken[res] = max(1, self.capacity(res)) if res in a.spec.exclusive else 1
                self._running[res] = self._running.get(res, 0) + a.taken[res]
            a.job.state = "running"
            a.job.waiting_for = None
            a.job.can_run_anyway = False
            threading.Thread(target=self._thread_main, args=(a,), name=f"job-{a.job.id}", daemon=True).start()

    def _thread_main(self, a: _Active) -> None:
        try:
            self._execute(a)
        finally:
            with self._cond:
                for res in a.spec.resources:
                    self._running[res] = max(0, self._running.get(res, 0) - a.taken.get(res, 1))
                self._cond.notify_all()

    def notify(self) -> None:
        """Wake the scheduler (a resource was freed outside a job)."""
        with self._cond:
            self._cond.notify_all()

    # -- adoption (micro-batching) ----------------------------------------------

    def _adopt(self, leader: _Active, limit: int) -> list[_Active]:
        group = leader.spec.group
        if not group or limit <= 0:
            return []
        with self._cond:
            found = sorted(
                (a for a in self._active.values() if a is not leader and a.spec.group == group and a.job.state in ("queued", "waiting") and not a.cancel.is_set()),
                key=lambda a: a.job.created_at,
            )[:limit]
            for a in found:
                a.adopted_by = leader.job.id
                a.job.state = "running"
                a.job.waiting_for = None
                a.job.can_run_anyway = False
                a.job.started_at = now_iso()
        for a in found:
            self._log(a, f"Running with job {leader.job.id} (one batch)")
            self._persist(a)
            self._publish(a.job)
        return found

    def _end_adopted(self, a: _Active, result: dict[str, Any] | None = None, error: BaseException | None = None) -> None:
        if a.cancel.is_set():
            self._finish(a, "cancelled")
        elif error is None:
            a.job.result = result or {}
            a.job.progress = 1.0
            self._finish(a, "completed")
        else:
            err = error if isinstance(error, MizuotoError) else MizuotoError(str(error) or type(error).__name__, {"exception": type(error).__name__})
            a.job.error = ErrorInfo(code=err.code, message=err.message, detail=err.detail)
            self._finish(a, "failed")

    def _release(self, a: _Active) -> None:
        with self._cond:
            if a.cancel.is_set():
                pass
            else:
                a.adopted_by = None
                a.job.state = "queued"
                a.job.started_at = None
                self._cond.notify_all()
        if a.cancel.is_set():
            self._finish(a, "cancelled")
            return
        self._persist(a)
        self._publish(a.job)

    # -- execution ------------------------------------------------------------

    def _execute(self, a: _Active) -> None:
        a.job.state = "running"
        a.job.started_at = now_iso()
        self._persist(a)
        self._publish(a.job)
        ctx = JobContext(self, a)
        try:
            if a.cancel.is_set():
                raise JobCancelled()
            result = a.spec.run(ctx)
            if a.cancel.is_set():
                raise JobCancelled()
            a.job.result = result or {}
            a.job.progress = 1.0
            self._finish(a, "completed")
        except (JobCancelled, CancelledError):
            self._finish(a, "cancelled")
        except Exception as e:
            if a.cancel.is_set():
                self._finish(a, "cancelled")
                return
            err = e if isinstance(e, MizuotoError) else None
            if err is None:
                self._log(a, traceback.format_exc())
                logger.error("Job %s failed: %s", a.job.id, e)
                err = MizuotoError(str(e) or type(e).__name__, {"exception": type(e).__name__})
            if isinstance(err, ComputeError) and err.detail.get("reason") == "out_of_memory":
                self._log(a, "Out of GPU memory: every model not in use was unloaded from the GPUs. Retry the job.")
                self._compute.release_after_oom(f"job {a.job.id} ({a.job.kind})", err.detail.get("device"))
            a.job.error = ErrorInfo(code=err.code, message=err.message, detail=err.detail)
            self._finish(a, "failed")

    def _finish(self, a: _Active, state: str) -> None:
        a.job.state = state  # ty: ignore[invalid-assignment]
        a.job.transfer = None
        a.job.finished_at = now_iso()
        a.job.can_retry = state in ("failed", "cancelled", "interrupted") and a.job.kind in self._factories
        a.job.can_run_anyway = False
        for s in a.job.steps:
            if s.state == "running":
                s.state = "cancelled" if state == "cancelled" else ("failed" if state == "failed" else "done")
            elif s.state == "pending" and state != "completed":
                s.state = "cancelled" if state == "cancelled" else "skipped"
        self._flush_log(a, force=True)
        self._persist(a)
        self._publish(a.job)
        with self._lock:
            self._active.pop(a.job.id, None)
        self._compute.emit_usage(force=True)

    def _run_worker(
        self, a: _Active, target: WorkerTarget, request: dict[str, Any], args: list[str], on_event: Callable[[Any], None] | None, heartbeat_timeout: float
    ) -> dict[str, Any]:
        from mizuoto.core.jobs.events import ErrorEvent, LogEvent, ProgressEvent, ResultEvent, StepEvent
        from mizuoto.core.jobs.jsonl import parse_lines

        self._logs.mkdir(parents=True, exist_ok=True)
        req_path = self._logs / f"{a.job.id}.{new_id()}.request.json"
        req_path.write_text(json.dumps(request), encoding="utf-8")
        result: dict[str, Any] = {}
        error: ErrorEvent | None = None
        last_seen = [time.monotonic()]
        hung = threading.Event()
        try:
            if a.cancel.is_set():
                raise JobCancelled()
            proc = start_target(target, *args, str(req_path))
            a.process = proc

            def watchdog() -> None:
                while proc.poll() is None:
                    if time.monotonic() - last_seen[0] > heartbeat_timeout:
                        hung.set()
                        kill_process_tree(proc)
                        return
                    time.sleep(min(1.0, heartbeat_timeout / 4))

            threading.Thread(target=watchdog, name=f"watchdog-{a.job.id}", daemon=True).start()
            assert proc.stdout is not None
            for line in proc.stdout:
                last_seen[0] = time.monotonic()
                for event in parse_lines(line):
                    if on_event is not None:
                        on_event(event)
                    if isinstance(event, ProgressEvent):
                        if on_event is None:
                            self._progress(a, event.value, event.step)
                            if event.message:
                                self._message(a, event.message)
                    elif isinstance(event, StepEvent):
                        if on_event is None:
                            self._step(a, event.id, event.state, event.progress)
                    elif isinstance(event, LogEvent):
                        self._log(a, event.message)
                    elif isinstance(event, ResultEvent):
                        result = event.data
                    elif isinstance(event, ErrorEvent):
                        error = event
                        self._log(a, f"error: {event.message}")
            code = proc.wait()
        finally:
            a.process = None
            req_path.unlink(missing_ok=True)
        if a.cancel.is_set():
            raise JobCancelled()
        if hung.is_set():
            raise OperationTimeoutError(f"The worker printed nothing for {heartbeat_timeout:.0f} s and was stopped", {"stage": target.module})
        if error is not None:
            raise error_from_code(error.code, error.message, error.detail)
        if code != 0:
            raise WorkerError(f"The worker process ended ({describe_exit(code)})", {"exit_code": code, "module": target.module})
        return result

    # -- reporting -------------------------------------------------------------

    def _progress(self, a: _Active, value: float | None, step: str | None) -> None:
        a.job.progress = None if value is None else max(0.0, min(1.0, float(value)))
        if step is not None:
            a.job.step = step
        now = time.monotonic()
        if now - a.last_progress_emit >= PROGRESS_INTERVAL:
            a.last_progress_emit = now
            self._publish(a.job)
        if now - a.last_db_write >= DB_PROGRESS_INTERVAL:
            self._persist(a)

    def _message(self, a: _Active, text: str | None) -> None:
        # A new stage is always told at once (only repeats are throttled): a throttled one stayed
        # unseen until the next update, which may be the job's end.
        changed = text != a.job.message
        a.job.message = text
        now = time.monotonic()
        if changed or now - a.last_progress_emit >= PROGRESS_INTERVAL:
            a.last_progress_emit = now
            self._publish(a.job)

    def _transfer(self, a: _Active, done: int, total: int) -> None:
        now = time.monotonic()
        samples = a.transfer_samples
        if samples and done < samples[-1][1]:
            samples.clear()  # a file started again from the beginning
        samples.append((now, done))
        while len(samples) > 2 and now - samples[0][0] > TRANSFER_WINDOW:
            samples.popleft()
        speed = eta = None
        (t0, d0), (t1, d1) = samples[0], samples[-1]
        if t1 - t0 >= 0.5:
            speed = max(0.0, (d1 - d0) / (t1 - t0))
            eta = (total - done) / speed if speed > 0 else None
        a.job.transfer = Transfer(done_bytes=done, total_bytes=total, bytes_per_second=speed, eta_seconds=eta)
        if now - a.last_progress_emit >= PROGRESS_INTERVAL:
            a.last_progress_emit = now
            self._publish(a.job)

    def _step(self, a: _Active, step_id: str, state: str, progress: float | None) -> None:
        for s in a.job.steps:
            if s.id == step_id:
                s.state = state  # ty: ignore[invalid-assignment]
                s.progress = progress if progress is not None else (1.0 if state == "done" else s.progress)
                break
        else:
            a.job.steps.append(JobStep(id=step_id, title=step_id, state=state, progress=progress))  # ty: ignore[invalid-argument-type]
        if state == "running":
            a.job.step = step_id
        self._persist(a)
        self._publish(a.job)

    def _add_steps(self, a: _Active, steps: list[JobStep]) -> None:
        known = {s.id for s in a.job.steps}
        a.job.steps.extend(s for s in steps if s.id not in known)
        self._persist(a)
        self._publish(a.job)

    def _log(self, a: _Active, text: str) -> None:
        self._logs.mkdir(parents=True, exist_ok=True)
        path = self._logs / f"{a.job.id}.log"
        lines = text.rstrip("\n").split("\n")
        with self._lock:
            offset = path.stat().st_size if path.exists() else 0
            with open(path, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
            if a.pending_offset is None:
                a.pending_offset = offset
            a.pending_lines.extend(lines)
        self._flush_log(a)

    def _flush_log(self, a: _Active, force: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            if not a.pending_lines or (not force and now - a.last_log_emit < LOG_EVENT_INTERVAL):
                return
            lines, offset = a.pending_lines[-MAX_LOG_EVENT_LINES:], a.pending_offset or 0
            a.pending_lines, a.pending_offset = [], None
            a.last_log_emit = now
        self._events.publish(JobLogEvent(job_id=a.job.id, offset=offset, lines=lines))

    def _persist(self, a: _Active) -> None:
        j = a.job
        a.last_db_write = time.monotonic()
        self._db.execute(
            "UPDATE jobs SET state = ?, title = ?, progress = ?, step = ?, steps = ?, error = ?, result = ?, started_at = ?, finished_at = ? WHERE id = ?",
            (
                j.state,
                j.title,
                j.progress,
                j.step,
                json.dumps([s.model_dump() for s in j.steps]),
                j.error.model_dump_json() if j.error else None,
                json.dumps(j.result) if j.result is not None else None,
                j.started_at,
                j.finished_at,
                j.id,
            ),
        )

    def _publish(self, job: Job) -> None:
        self._events.publish(JobUpdatedEvent(job=job.model_copy(deep=True)))

    # -- control ---------------------------------------------------------------

    def wait(self, job_id: str, timeout: float | None = None) -> Job:
        """The job once it has finished, or as it is when ``timeout`` seconds pass."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                a = self._active.get(job_id)
                if a is None:
                    break
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    return a.job.model_copy(deep=True)
                self._cond.wait(min(0.5, left) if left is not None else 0.5)
        return self.get(job_id)

    def cancel(self, job_id: str) -> Job:
        with self._cond:
            a = self._active.get(job_id)
            if a is None:
                job = self.get(job_id)
                if job.state in FINISHED_STATES:
                    return job
                raise NotFoundError(f"Job {job_id} is not running")
            a.cancel.set()
            if a.job.state in ("queued", "waiting"):
                self._active.pop(job_id, None)
                a.job.state = "cancelled"
                a.job.finished_at = now_iso()
                a.job.can_retry = a.job.kind in self._factories
                self._persist(a)
                self._publish(a.job)
                return a.job.model_copy(deep=True)
            a.job.state = "cancelling"
            proc = a.process
        self._publish(a.job)
        if proc is not None:
            threading.Thread(target=kill_process_tree, args=(proc,), daemon=True).start()
        return a.job.model_copy(deep=True)

    def run_anyway(self, job_id: str) -> Job:
        """Start a job that waits for a GPU slot without waiting (the user knows the memory suffices)."""
        with self._cond:
            a = self._active.get(job_id)
            if a is None or a.job.state not in ("queued", "waiting"):
                raise ConflictError("Only a waiting job can be started anyway")
            a.run_anyway = True
            self._cond.notify_all()
            return a.job.model_copy(deep=True)

    def retry(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.state not in ("failed", "cancelled", "interrupted"):
            raise ConflictError("Only a failed, cancelled or interrupted job can be retried")
        return self.submit(job.kind, job.request, job.origin)

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job.state in ACTIVE_STATES:
            raise ConflictError("Cancel the job before removing it")
        self._db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        self._remove_log(job_id)
        self._events.publish(JobsClearedEvent(ids=[job_id]))

    def clear_finished(self) -> list[str]:
        rows = self._db.fetchall(f"SELECT id FROM jobs WHERE state IN ({','.join('?' * len(FINISHED_STATES))})", tuple(FINISHED_STATES))
        ids = [r["id"] for r in rows]
        if ids:
            with self._db.transaction() as conn:
                conn.executemany("DELETE FROM jobs WHERE id = ?", [(i,) for i in ids])
            for i in ids:
                self._remove_log(i)
            self._events.publish(JobsClearedEvent(ids=ids))
        return ids

    def _remove_log(self, job_id: str) -> None:
        (self._logs / f"{job_id}.log").unlink(missing_ok=True)

    # -- reading ---------------------------------------------------------------

    def get(self, job_id: str) -> Job:
        with self._lock:
            a = self._active.get(job_id)
            if a is not None:
                return a.job.model_copy(deep=True)
        row = self._db.fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if row is None:
            raise NotFoundError(f"Unknown job: {job_id}")
        return self._row(row)

    def list_jobs(self, state: str | None = None, kind: str | None = None, cursor: str | None = None, limit: int = 50) -> JobPage:
        where, params = [], []
        if state == "active":
            where.append(f"state IN ({','.join('?' * len(ACTIVE_STATES))})")
            params += sorted(ACTIVE_STATES)
        elif state == "finished":
            where.append(f"state IN ({','.join('?' * len(FINISHED_STATES))})")
            params += sorted(FINISHED_STATES)
        elif state:
            where.append("state = ?")
            params.append(state)
        if kind:
            where.append("kind = ?")
            params.append(kind)
        if cursor:
            created, _, cid = cursor.partition("|")
            where.append("(created_at < ? OR (created_at = ? AND id < ?))")
            params += [created, created, cid]
        sql = "SELECT * FROM jobs" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created_at DESC, id DESC LIMIT ?"
        rows = self._db.fetchall(sql, (*params, limit + 1))
        items = []
        with self._lock:
            for r in rows[:limit]:
                a = self._active.get(r["id"])
                items.append(a.job.model_copy(deep=True) if a else self._row(r))
        next_cursor = f"{rows[limit - 1]['created_at']}|{rows[limit - 1]['id']}" if len(rows) > limit else None
        return JobPage(items=items, next_cursor=next_cursor)

    def active(self) -> list[Job]:
        with self._lock:
            return [a.job.model_copy(deep=True) for a in self._active.values()]

    def read_log(self, job_id: str, offset: int = 0, limit_bytes: int = 256 * 1024) -> JobLog:
        self.get(job_id)
        path = self._logs / f"{job_id}.log"
        if not path.exists():
            return JobLog(job_id=job_id, offset=offset, next_offset=offset, lines=[])
        with open(path, "rb") as f:
            f.seek(max(0, offset))
            data = f.read(limit_bytes)
        if len(data) == limit_bytes and b"\n" in data:
            data = data[: data.rindex(b"\n") + 1]
        text = data.decode("utf-8", errors="replace")
        return JobLog(job_id=job_id, offset=offset, next_offset=offset + len(data), lines=text.splitlines())

    def _row(self, r: Any) -> Job:
        job = Job(
            id=r["id"],
            kind=r["kind"],
            title=r["title"],
            state=r["state"],
            progress=r["progress"],
            step=r["step"],
            steps=[JobStep.model_validate(s) for s in json.loads(r["steps"] or "[]")],
            error=ErrorInfo.model_validate_json(r["error"]) if r["error"] else None,
            result=json.loads(r["result"]) if r["result"] else None,
            request=json.loads(r["request"]),
            origin=r["origin"],
            created_at=r["created_at"],
            started_at=r["started_at"],
            finished_at=r["finished_at"],
        )
        job.can_retry = job.state in ("failed", "cancelled", "interrupted") and job.kind in self._factories
        return job

    def _mark_interrupted(self) -> None:
        """A job still active whose process has ended is ``interrupted`` now; another live process's jobs
        (a server, a command line rendering a book) are left alone."""
        from mizuoto.core.net.runtime_file import pid_alive

        states = sorted(ACTIVE_STATES)
        rows = self._db.fetchall(f"SELECT id, owner_pid FROM jobs WHERE state IN ({','.join('?' * len(states))})", tuple(states))
        gone = [r["id"] for r in rows if not (r["owner_pid"] and r["owner_pid"] != os.getpid() and pid_alive(int(r["owner_pid"])))]
        for job_id in gone:
            self._db.execute("UPDATE jobs SET state = 'interrupted', finished_at = ? WHERE id = ?", (now_iso(), job_id))
