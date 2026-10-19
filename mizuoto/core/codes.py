"""Error codes, reasons and exit codes: standard library only.

``mizuoto.core.errors`` maps each code to its exception class (``error_from_code``). The runtime, the engines
and job workers (training stages) need only the strings, so they never import the core's pydantic models.
"""

INTERNAL_ERROR = "internal_error"
NOT_FOUND = "not_found"
CONFLICT = "conflict"
BUSY = "busy"
INVALID_PATH = "invalid_path"
INVALID_INPUT = "invalid_input"
CAPABILITY_NOT_SUPPORTED = "capability_not_supported"
PARAMETER_NOT_SUPPORTED = "parameter_not_supported"
PARAMETER_OUT_OF_RANGE = "parameter_out_of_range"
REFERENCE_INVALID = "reference_invalid"
LANGUAGE_NOT_SUPPORTED = "language_not_supported"
LICENCE_NOT_ACCEPTED = "licence_not_accepted"
MODEL_MISSING = "model_missing"
ENGINE_NOT_INSTALLED = "engine_not_installed"
ASSET_MISSING = "asset_missing"
DEVICE_UNAVAILABLE = "device_unavailable"
COMPUTE_UNAVAILABLE = "compute_unavailable"
WORKER_FAILED = "worker_failed"
ENGINE_ERROR = "engine_error"
TIMEOUT = "timeout"
CANCELLED = "cancelled"
NOT_LOCAL = "not_local"
DOWNLOAD_FAILED = "download_failed"
INSTALL_FAILED = "install_failed"
REMOTE_UNAVAILABLE = "remote_unavailable"

# detail["reason"] values of not_found
FILE_MISSING = "file_missing"
"""A record's file is neither where it was recorded nor anywhere it could have moved to."""

# detail["reason"] values of compute_unavailable
OUT_OF_MEMORY = "out_of_memory"
NO_DEVICE = "no_device"

# Process exit codes of job workers
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_RETRY = 75
"""The worker asks to be run again (for example in fp32 after an fp16 failure)."""
EXIT_CANCELLED = 130
