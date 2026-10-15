"""Add the event models to the OpenAPI schema, so the web UI types socket payloads from it."""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from mizuoto.core.events.models import EventBase


class MizuotoAPI(FastAPI):
    """FastAPI with socket event models included in its OpenAPI schema."""

    def openapi(self) -> dict[str, Any]:
        if self.openapi_schema:
            return self.openapi_schema
        schema = get_openapi(title=self.title, version=self.version, routes=self.routes, description=self.description)
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        events: dict[str, dict[str, str]] = {}
        for event_cls in EventBase.get_events():
            model_schema = event_cls.model_json_schema(ref_template="#/components/schemas/{model}", mode="serialization")
            for name, sub in model_schema.pop("$defs", {}).items():
                components.setdefault(name, sub)
            components[event_cls.__name__] = model_schema
            events[event_cls.__event_name__] = {"$ref": f"#/components/schemas/{event_cls.__name__}"}
        components["ServerEvents"] = {"type": "object", "properties": events, "required": sorted(events)}
        schema["components"]["schemas"] = dict(sorted(components.items()))
        self.openapi_schema = schema
        return schema
