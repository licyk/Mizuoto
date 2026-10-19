"""Checking common parameters and engine options against a manifest.

``None`` means "the engine's default": only the parameters a request sets are checked and sent.
A parameter the model does not take is an error in strict mode and a warning otherwise; a value out
of range is always an error. Defaults are never filled in here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mizuoto.core.engines.models import EngineManifest, OptionSpec, ParamField, ParamSpec
from mizuoto.core.errors import ParameterError


def _in_variant(spec: ParamSpec | OptionSpec, variant: str) -> bool:
    """Whether a parameter applies to ``variant`` (empty: any variant, as for an engine's load options)."""
    return spec.variants is None or not variant or variant in spec.variants


def accepted(specs: Mapping[Any, ParamSpec | OptionSpec], variant: str) -> frozenset[str]:
    """The names of ``specs`` (a manifest's parameters or options) that ``variant`` takes, offered or not."""
    return frozenset(name for name, spec in specs.items() if _in_variant(spec, variant))


def _check_number(name: str, value: Any, kind: str, lo: float | None, hi: float | None) -> tuple[Any, dict[str, Any] | None]:
    if kind == "bool":
        if isinstance(value, bool):
            return value, None
        return value, {"name": name, "reason": "not_a_boolean"}
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value, {"name": name, "reason": "not_a_number"}
    if kind == "int":
        if isinstance(value, float) and not value.is_integer():
            return value, {"name": name, "reason": "not_an_integer"}
        value = int(value)
    else:
        value = float(value)
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        return value, {"name": name, "reason": "out_of_range", "min": lo, "max": hi, "value": value}
    return value, None


def check_params(manifest: EngineManifest, variant: str, params: dict[str, Any], *, strict: bool, streaming: bool = False) -> tuple[dict[str, Any], list[str]]:
    """The parameters to send, and warnings. Raises ``ParameterError`` for what cannot be sent."""
    accepted: dict[str, Any] = {}
    warnings: list[str] = []
    unsupported: list[dict[str, Any]] = []
    out_of_range: list[dict[str, Any]] = []
    for name, value in params.items():
        if value is None:
            continue
        spec = manifest.params.get(name)
        if spec is None or not _in_variant(spec, variant):
            unsupported.append({"name": name, "reason": "not_supported"})
            continue
        if streaming and not spec.streaming:
            unsupported.append({"name": name, "reason": "not_while_streaming"})
            continue
        value, problem = _check_number(name, value, spec.type, spec.min, spec.max)
        if problem is not None:
            out_of_range.append(problem)
            continue
        if spec.approximate:
            warnings.append(f"{name} is approximated by the engine's {spec.target or name}")
        accepted[name] = value
    if out_of_range:
        raise ParameterError(f"Invalid parameter values: {', '.join(f['name'] for f in out_of_range)}", out_of_range, out_of_range=True)
    if unsupported:
        names = ", ".join(f["name"] for f in unsupported)
        if strict:
            supported = sorted(k for k, s in manifest.params.items() if _in_variant(s, variant))
            raise ParameterError(f"The model does not take: {names}", unsupported, {"supported": supported})
        warnings.extend(f"Ignored {f['name']}: {f['reason'].replace('_', ' ')}" for f in unsupported)
    return accepted, warnings


def check_options(manifest: EngineManifest, variant: str, options: dict[str, Any], schema: dict[str, OptionSpec] | None = None) -> dict[str, Any]:
    """Engine options as given, checked against the manifest. Unknown options are always an error:
    an engine option is specific by definition, so a misspelt one must not pass silently."""
    schema = manifest.options if schema is None else schema
    problems: list[dict[str, Any]] = []
    out: dict[str, Any] = {}
    for name, value in options.items():
        spec = schema.get(name)
        if spec is None or not _in_variant(spec, variant):
            problems.append({"name": name, "reason": "unknown_option"})
            continue
        if value is None:
            continue
        if spec.type == "enum":
            if value not in (spec.choices or []):
                problems.append({"name": name, "reason": "not_a_choice", "choices": spec.choices, "value": value})
                continue
        elif spec.type == "str":
            if not isinstance(value, str):
                problems.append({"name": name, "reason": "not_a_string"})
                continue
        else:
            value, problem = _check_number(name, value, spec.type, spec.min, spec.max)
            if problem is not None:
                problems.append(problem)
                continue
        out[name] = value
    if problems:
        raise ParameterError(f"Invalid engine options: {', '.join(p['name'] for p in problems)}", problems, {"options": sorted(schema)})
    return out


def param_fields(manifest: EngineManifest, variant: str) -> list[ParamField]:
    """The common parameters, then the options, of one variant, for the UI's form."""
    fields: list[ParamField] = []
    for name, spec in manifest.params.items():
        if _in_variant(spec, variant) and not spec.hidden:
            fields.append(
                ParamField(
                    name=name,
                    kind="common",
                    type=spec.type,
                    default=spec.default,
                    min=spec.min,
                    max=spec.max,
                    step=spec.step,
                    group="basic",
                    help=spec.help,
                    approximate=spec.approximate,
                    streaming=spec.streaming,
                )
            )
    for name, spec in manifest.options.items():
        if _in_variant(spec, variant) and spec.group != "hidden":
            fields.append(
                ParamField(
                    name=name,
                    kind="option",
                    type=spec.type,
                    default=spec.default,
                    min=spec.min,
                    max=spec.max,
                    step=spec.step,
                    choices=spec.choices,
                    group=spec.group,
                    help=spec.help,
                )
            )
    return fields
