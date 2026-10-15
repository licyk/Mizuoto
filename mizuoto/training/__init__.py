"""Native training: an engine's trainer runs each stage of a run in a child process of
the host's own interpreter, on the native modules.

``python -m mizuoto.training.stage <request.json>`` runs one stage: the
request is the one ``TrainingService`` builds for every trainer (the rows, the base model, the hyperparameters, the
earlier stages' results); the trainer reports through ``StageContext`` (progress, metrics, files, log lines,
previews) as job-worker events. The process gives back the memory after a run and contains crashes; that is why
stages are not run in the host's own process.
"""

from mizuoto.training.context import Row, StageContext
from mizuoto.training.trainer import Trainer, load_trainer, trainer_for

__all__ = ["Row", "StageContext", "Trainer", "load_trainer", "trainer_for"]
