"""The native trainer contract: one class per engine, a method per stage its manifest declares."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from mizuoto.core import codes

if TYPE_CHECKING:
    from mizuoto.training.context import StageContext


class Trainer:
    """An engine's training: ``stage_<id>(ctx)`` for each stage of its methods, run one per process."""

    ENGINE: ClassVar[str] = ""
    REQUIRES: ClassVar[tuple[str, ...]] = ()
    """Modules training needs beyond inference (none of the engines has any today), probed by import before a run."""
    DEVICES: ClassVar[tuple[str, ...] | None] = None
    """The device kinds the trainer runs on, when they differ from its manifest method's ``devices`` (these win)."""

    @classmethod
    def refuses(cls, method: str, model_dir: Path) -> str | None:
        """Why ``method`` cannot start from the model in ``model_dir`` (read without torch: shown before a run, where
        the stage would fail), else None."""
        return None

    def run_stage(self, stage: str, ctx: StageContext) -> dict[str, Any]:
        from mizuoto.runtime.engine import EngineError

        method = getattr(self, f"stage_{stage}", None)
        if method is None:
            raise EngineError(f"{self.ENGINE} has no training stage {stage!r}", code=codes.INVALID_INPUT)
        return method(ctx) or {}


def trainer_for(engine_id: str) -> str | None:
    """``module:Class`` of the engine's native trainer, or ``None`` (loads no torch: engine modules import it lazily)."""
    from mizuoto.runtime.engine import engine_ids, get_engine

    if engine_id not in engine_ids():
        return None
    return type(get_engine(engine_id)).TRAINER


def load_trainer(spec: str) -> Trainer:
    module_name, _, class_name = spec.partition(":")
    trainer = getattr(importlib.import_module(module_name), class_name)()
    if not isinstance(trainer, Trainer):
        raise TypeError(f"{spec} is not a mizuoto.training.Trainer")
    return trainer


def trainer_class(spec: str) -> type[Trainer]:
    """The trainer's class (its module imports without torch, so this is cheap)."""
    module_name, _, class_name = spec.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


def missing_requirements(spec: str) -> list[str]:
    """The trainer's ``REQUIRES`` that do not import here (checked without importing the trainer's heavy modules)."""
    import importlib.util

    return [m for m in trainer_class(spec).REQUIRES if importlib.util.find_spec(m) is None]
