"""Load, override and save settings."""

import copy
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import ValidationError as PydanticValidationError

from mizuoto.core.errors import ValidationError
from mizuoto.core.files import volume_roots
from mizuoto.core.paths import CONFIG_DIR_ENV, DATA_DIR_ENV, ENV_PREFIX, default_data_dir
from mizuoto.core.settings.models import AgentSettingsView, EngineSettings, RemoteHostView, RemoteSettingsView, ResolvedPaths, ServerSettingsView, Settings, SettingsView

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

SETTINGS_FILE_NAME = "settings.toml"
FILE_CHECK_INTERVAL = 1.0
"""Seconds between checks of ``settings.toml`` for changes made outside this process."""

logger = logging.getLogger(__name__)


def changed_keys(old: Any, new: Any, prefix: str = "") -> list[str]:
    """Dotted names of the leaves that differ between two nested dicts."""
    if isinstance(old, dict) and isinstance(new, dict):
        out: list[str] = []
        for key in sorted(set(old) | set(new)):
            out += changed_keys(old.get(key), new.get(key), f"{prefix}{key}.")
        return out
    return [] if old == new else [prefix.rstrip(".")]


def _parse_scalar(raw: str) -> Any:
    """Turn a command-line or environment string into a value: JSON when it parses, else the string."""
    text = raw.strip()
    if text.lower() in ("null", "none"):
        return None
    try:
        return json.loads(text)
    except ValueError:
        return raw


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _strip_none(value: Any) -> Any:
    """TOML has no null: drop keys whose value is None."""
    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(v) for v in value]
    return value


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def _get_dotted(data: Any, dotted: str) -> Any:
    node = data
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise KeyError(dotted)
    return node


class SettingsService:
    """Owns ``settings.toml``. Environment variables with the ``MIZUOTO_`` prefix override it.

    An override uses ``__`` between levels, for example ``MIZUOTO_SERVER__PORT=8000``.
    Overrides are applied on top of the file and are never written back to it.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        settings_path: Path | None = None,
        environ: Mapping[str, str] | None = None,
        overrides: dict[str, Any] | None = None,
        config_dir: Path | None = None,
    ) -> None:
        """The settings file is ``settings_path``, else ``settings.toml`` in ``config_dir`` (or
        ``MIZUOTO_CONFIG_DIR``), else in the data directory.

        ``overrides`` come from a host application embedding this package.

        They sit above the file and the environment, are never written back, and cannot be
        changed through the API, so a host can pin a value for the life of the process.
        """
        self._environ: Mapping[str, str] = environ if environ is not None else os.environ
        self.data_dir = (data_dir or default_data_dir(self._environ)).resolve()
        if config_dir is None and self._environ.get(CONFIG_DIR_ENV):
            config_dir = Path(self._environ[CONFIG_DIR_ENV])
        self.config_dir = config_dir.expanduser().resolve() if config_dir else self.data_dir
        self.path = settings_path.expanduser().resolve() if settings_path else self.config_dir / SETTINGS_FILE_NAME
        self._overrides = copy.deepcopy(overrides or {})
        self._lock = threading.RLock()
        self._listeners: list[Callable[[Settings], None]] = []
        self._key_listeners: list[Callable[[list[str]], None]] = []
        self._file_data: dict[str, Any] = {}
        self._file_stamp: tuple[int, int] | None = None
        self._checked = time.monotonic()
        self._settings = Settings()
        self.reload()

    # -- loading -----------------------------------------------------------

    def env_overrides(self) -> dict[str, Any]:
        """Return the overrides found in the environment, as a nested dict."""
        out: dict[str, Any] = {}
        for key, raw in self._environ.items():
            if not key.startswith(ENV_PREFIX) or key in (DATA_DIR_ENV, CONFIG_DIR_ENV) or "__" not in key:
                continue
            dotted = key[len(ENV_PREFIX) :].lower().replace("__", ".")
            _set_dotted(out, dotted, _parse_scalar(raw))
        return out

    def env_override_names(self) -> list[str]:
        return sorted(k for k in self._environ if k.startswith(ENV_PREFIX) and k not in (DATA_DIR_ENV, CONFIG_DIR_ENV) and "__" in k)

    def pinned_names(self) -> list[str]:
        """Dotted names of every value the host application pinned, such as ``server.port``."""

        def walk(data: dict[str, Any], prefix: str) -> list[str]:
            names: list[str] = []
            for key, value in data.items():
                name = f"{prefix}{key}"
                names += walk(value, f"{name}.") if isinstance(value, dict) and value else [name]
            return names

        return sorted(walk(self._overrides, ""))

    def reload(self) -> None:
        with self._lock:
            stamp = self._stamp()
            if self.path.is_file():
                with open(self.path, "rb") as f:
                    file_data = tomllib.load(f)
            else:
                file_data = {}
            self._settings = self._validate(self._effective(file_data))
            self._file_data = file_data
            self._file_stamp = stamp

    def _stamp(self) -> tuple[int, int] | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return st.st_mtime_ns, st.st_size

    def _refresh(self, force: bool = False) -> tuple[Settings, Settings] | None:
        """Re-read ``settings.toml`` when another process (``mizuoto config set``, an editor) changed
        it; checked at most every ``FILE_CHECK_INTERVAL`` seconds unless ``force``. A file that does not
        parse or validate is reported and the settings in use are kept. Returns (old, new) on a change.
        """
        with self._lock:
            now = time.monotonic()
            if not force and now - self._checked < FILE_CHECK_INTERVAL:
                return None
            self._checked = now
            stamp = self._stamp()
            if stamp == self._file_stamp:
                return None
            old = self._settings
            try:
                self.reload()
            except (ValidationError, tomllib.TOMLDecodeError, OSError) as e:
                logger.warning("Ignored a change to %s: %s", self.path, e)
                self._file_stamp = stamp
                return None
            return old, self._settings

    def _notify(self, old: Settings, new: Settings) -> None:
        keys = changed_keys(old.model_dump(), new.model_dump())
        if not keys:
            return
        for listener in self._listeners:
            listener(new)
        for key_listener in self._key_listeners:
            key_listener(keys)

    def _effective(self, file_data: dict[str, Any]) -> dict[str, Any]:
        """Defaults, then the file, then the environment, then the host application's overrides."""
        merged = _deep_merge(Settings().model_dump(), _deep_merge(file_data, self.env_overrides()))
        return _deep_merge(merged, self._overrides)

    @staticmethod
    def _validate(data: dict[str, Any]) -> Settings:
        try:
            return Settings.model_validate(data)
        except PydanticValidationError as e:
            errors = [{key: value for key, value in error.items() if key != "url"} for error in e.errors()]
            first = e.errors()[0]
            raise ValidationError(f"Invalid settings: {first['msg']} at {'.'.join(str(p) for p in first['loc'])}", {"errors": errors}) from e

    @property
    def settings(self) -> Settings:
        """The effective settings: defaults, then the file, then the environment.

        A change another process made to the file is picked up within ``FILE_CHECK_INTERVAL``.
        """
        changed = self._refresh()
        if changed:
            self._notify(*changed)
        return self._settings

    def on_change(self, listener: Callable[[Settings], None]) -> None:
        """Called with the new settings after a change (from this process or from the file)."""
        self._listeners.append(listener)

    def on_keys_change(self, listener: Callable[[list[str]], None]) -> None:
        """Called with the dotted names that changed, such as ``["paths.models_dir"]``."""
        self._key_listeners.append(listener)

    # -- saving ------------------------------------------------------------

    def update(self, patch: dict[str, Any]) -> Settings:
        """Deep-merge ``patch`` into the file settings, validate, save and return the effective settings.

        Dicts merge; every other value replaces. ``None`` clears an optional value, such as a token.
        """
        with self._lock:
            # Start from the file as it is now, so a change another process made is kept, not overwritten.
            outside = self._refresh(force=True)
            before = outside[0] if outside else self._settings
            # A None in the patch clears the key, now (as the file, which cannot hold None, will on its next read).
            merged_file = _strip_none(_deep_merge(self._file_data, patch))
            file_settings = self._validate(_deep_merge(Settings().model_dump(), merged_file))
            effective = self._validate(self._effective(file_settings.model_dump()))
            self._write(file_settings)
            self._file_data = file_settings.model_dump()
            self._settings = effective
        self._notify(before, effective)
        return effective

    def mutate(self, fn: Callable[[dict[str, Any]], None]) -> Settings:
        """Apply ``fn`` to a copy of the file settings as a dict, then save."""
        changed = self._refresh(force=True)
        if changed:
            self._notify(*changed)
        with self._lock:
            data = _deep_merge(Settings().model_dump(), self._file_data)
            fn(data)
            return self.update(data)

    def _write(self, settings: Settings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".toml.tmp")
        with open(tmp, "wb") as f:
            tomli_w.dump(_strip_none(settings.model_dump()), f)
        os.replace(tmp, self.path)
        self._file_stamp = self._stamp()

    # -- dotted access for the command line --------------------------------

    def get_value(self, dotted: str) -> Any:
        try:
            value = _get_dotted(self.view().model_dump(), dotted)
        except KeyError:
            raise ValidationError(f"Unknown setting: {dotted}") from None
        return value

    def set_value(self, dotted: str, raw: str) -> Settings:
        try:
            _get_dotted(self.settings.model_dump(), dotted)
        except KeyError:
            # Allow new keys only inside dict-valued groups, and the fields of an engine not yet configured.
            parent = dotted.rsplit(".", 1)[0]
            parts = dotted.split(".")
            if not (len(parts) >= 3 and parts[0] == "engines" and parts[2] in EngineSettings.model_fields and (len(parts) == 3 or parts[2] == "load_options")):
                try:
                    if not isinstance(_get_dotted(self.settings.model_dump(), parent), dict):
                        raise KeyError(parent)
                except KeyError:
                    raise ValidationError(f"Unknown setting: {dotted}") from None
        patch: dict[str, Any] = {}
        _set_dotted(patch, dotted, _parse_scalar(raw))
        return self.update(patch)

    # -- resolved folders ---------------------------------------------------

    def _folder(self, value: str, default: str) -> Path:
        """A ``paths.*`` folder: empty is ``default`` inside the data folder, and a relative one is inside the data
        folder too (never the working directory, which changes with every launch)."""
        return (self.data_dir / Path(value).expanduser()).resolve() if value else self.data_dir / default

    @property
    def models_dir(self) -> Path:
        return self._folder(self.settings.paths.models_dir, "models")

    @property
    def assets_dir(self) -> Path:
        """Files models of one engine share; beside the models, so whoever shares the models shares them too."""
        return self.models_dir / "_assets"

    @property
    def voices_dir(self) -> Path:
        return self._folder(self.settings.paths.voices_dir, "voices")

    @property
    def outputs_dir(self) -> Path:
        return self._folder(self.settings.paths.outputs_dir, "outputs")

    @property
    def datasets_dir(self) -> Path:
        return self._folder(self.settings.paths.datasets_dir, "datasets")

    @property
    def experiments_dir(self) -> Path:
        return self._folder(self.settings.paths.experiments_dir, "experiments")

    @property
    def projects_dir(self) -> Path:
        return self._folder(self.settings.paths.projects_dir, "projects")

    @property
    def cache_dir(self) -> Path:
        return self._folder(self.settings.paths.cache_dir, "cache")

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    def browse_roots(self) -> list[Path]:
        """Server folders the UI may read: the configured ones, else home, the data directory and every
        drive (Windows) or mounted volume (macOS, Linux)."""
        paths = self.settings.paths
        if paths.browse_roots:
            roots = [Path(p).expanduser().resolve() for p in paths.browse_roots]
        else:
            # The volumes are absolute already and are not resolved: resolving a drive opens it.
            roots = [Path.home().resolve(), self.data_dir, *volume_roots()]
        out: list[Path] = []
        for root in roots:
            if root not in out:
                out.append(root)
        return out

    def resolved_paths(self) -> ResolvedPaths:
        return ResolvedPaths(
            models_dir=str(self.models_dir),
            voices_dir=str(self.voices_dir),
            outputs_dir=str(self.outputs_dir),
            datasets_dir=str(self.datasets_dir),
            experiments_dir=str(self.experiments_dir),
            projects_dir=str(self.projects_dir),
            cache_dir=str(self.cache_dir),
            browse_roots=[str(p) for p in self.browse_roots()],
        )

    # -- public view -------------------------------------------------------

    def view(self) -> SettingsView:
        s = self.settings
        return SettingsView(
            data_dir=str(self.data_dir),
            settings_file=str(self.path),
            server=ServerSettingsView(
                host=s.server.host,
                port=s.server.port,
                strict_port=s.server.strict_port,
                open_browser=s.server.open_browser,
                access_token_configured=bool(s.server.access_token),
                allowed_origins=s.server.allowed_origins,
                ssl=bool(s.server.ssl_certfile),
            ),
            paths=s.paths,
            resolved_paths=self.resolved_paths(),
            compute=s.compute,
            downloads=s.downloads,
            tts=s.tts,
            engines=s.engines,
            training=s.training,
            outputs=s.outputs,
            remote=RemoteSettingsView(
                hosts={n: RemoteHostView(address=h.address, token_configured=bool(h.token), tls=h.tls, ca_file=h.ca_file) for n, h in s.remote.hosts.items()},
                agent=AgentSettingsView(host=s.remote.agent.host, port=s.remote.agent.port, token_configured=bool(s.remote.agent.token), ssl=bool(s.remote.agent.ssl_certfile)),
                connect_timeout=s.remote.connect_timeout,
                heartbeat_timeout=s.remote.heartbeat_timeout,
            ),
            env_overrides=self.env_override_names(),
            pinned=self.pinned_names(),
        )
