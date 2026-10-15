"""Base class for records that cross the API boundary."""

from pydantic import BaseModel, ConfigDict


class Record(BaseModel):
    """Every model the API sends or receives, and every ``--json`` output.

    Defaulted fields are always present in output, so the schema marks them required and the
    generated TypeScript types match what the server sends.
    """

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)
