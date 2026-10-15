"""Model records."""

from typing import Literal

from pydantic import Field

from mizuoto.core.record import Record

ModelState = Literal["builtin", "installed", "missing", "partial", "corrupt", "downloading"]


class ModelInfo(Record):
    id: str
    engine: str
    variant: str
    title: str
    description: str = ""
    state: ModelState
    path: str | None = None
    """The model's folder, when installed."""
    source: Literal["catalog", "user"] = "catalog"
    size_mb: float | None = None
    vram_mb: float | None = None
    languages: list[str] | None = None
    recommended: bool = False
    capabilities: list[str] = Field(default_factory=list)
    """Its variant's capabilities (from the manifest)."""
    assets: list[str] = Field(default_factory=list)
    """Shared assets it needs; a model is installed only when they are too."""
    remote: str | None = None
    """The remote host its engine runs on; ``state`` is then the agent's."""


class AssetInfo(Record):
    """Files several models of an engine share, downloaded once (``<models>/_assets/<engine>/<name>/``)."""

    id: str
    engine: str
    title: str
    description: str = ""
    state: ModelState
    path: str | None = None
    size_mb: float | None = None
    used_by: list[str] = Field(default_factory=list)
    """The catalogue models that need it."""


class VerifyResult(Record):
    model: str
    ok: bool
    checked: int
    """Files hashed."""
    bad: list[str] = Field(default_factory=list)
    """Files missing or with the wrong size or SHA-256."""


class Detected(Record):
    """What a folder of model files is, judged from its contents."""

    engine: str
    variant: str
    title: str
    catalog_id: str | None = None
    """The catalogue model with the same files, when there is one."""
