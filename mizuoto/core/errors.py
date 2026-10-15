"""Domain exceptions shared by the API and the command line.

Each class carries ``code``, ``http_status`` and ``exit_code``, so both wrappers translate
mechanically. The code strings live in :mod:`mizuoto.core.codes` (standard library only), shared with the runtime and job workers.
"""

from typing import Any

from mizuoto.core import codes


class MizuotoError(Exception):
    """Base class for every error the wrappers translate."""

    code = codes.INTERNAL_ERROR
    http_status = 500
    exit_code = 1

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class NotFoundError(MizuotoError):
    """A model, voice, job, output or engine is unknown."""

    code = codes.NOT_FOUND
    http_status = 404
    exit_code = 2


class ConflictError(MizuotoError):
    """A name clash, or an operation that collides with one in progress."""

    code = codes.CONFLICT
    http_status = 409
    exit_code = 3


class BusyError(ConflictError):
    """The resource is in use (a worker, an environment being installed, a running stage)."""

    code = codes.BUSY


class InvalidPathError(MizuotoError):
    """A path escapes its root, or a server path is outside the browse roots."""

    code = codes.INVALID_PATH
    http_status = 400
    exit_code = 4


class ValidationError(MizuotoError):
    """Input that does not validate."""

    code = codes.INVALID_INPUT
    http_status = 400
    exit_code = 4


class CapabilityError(ValidationError):
    """The model cannot do what was asked (voice design on a clone-only model, streaming text, …)."""

    code = codes.CAPABILITY_NOT_SUPPORTED
    http_status = 422


class ParameterError(ValidationError):
    """A parameter the model does not take, or a value out of its range. ``detail["fields"]`` lists them."""

    code = codes.PARAMETER_NOT_SUPPORTED
    http_status = 422

    def __init__(self, message: str, fields: list[dict[str, Any]], detail: dict[str, Any] | None = None, *, out_of_range: bool = False) -> None:
        super().__init__(message, {**(detail or {}), "fields": fields})
        if out_of_range:
            self.code = codes.PARAMETER_OUT_OF_RANGE


class ReferenceInvalidError(ValidationError):
    """A reference clip the model cannot use: too short, too long, too low a sample rate, or silent."""

    code = codes.REFERENCE_INVALID
    http_status = 422


class LanguageError(ValidationError):
    """A language the model does not speak."""

    code = codes.LANGUAGE_NOT_SUPPORTED
    http_status = 422


class LicenceError(MizuotoError):
    """The engine's licence has not been accepted."""

    code = codes.LICENCE_NOT_ACCEPTED
    http_status = 403
    exit_code = 5


class ModelMissingError(MizuotoError):
    """A model is not downloaded. ``detail["models"]`` lists ``{id, size}``."""

    code = codes.MODEL_MISSING
    http_status = 409
    exit_code = 7


class EnvMissingError(MizuotoError):
    """A package an engine needs does not import here (torch, a licence-gated engine's distribution). ``detail["missing"]``
    lists them."""

    code = codes.ENGINE_NOT_INSTALLED
    http_status = 409
    exit_code = 7


class AssetMissingError(MizuotoError):
    """A shared asset is not installed. ``detail["assets"]`` lists the ids."""

    code = codes.ASSET_MISSING
    http_status = 409
    exit_code = 7

    def __init__(self, message: str, assets: list[str], detail: dict[str, Any] | None = None) -> None:
        super().__init__(message, {**(detail or {}), "assets": list(assets)})
        self.assets = list(assets)


class DeviceError(MizuotoError):
    """A device is missing or unusable. ``detail["reason"]`` says why."""

    code = codes.DEVICE_UNAVAILABLE
    http_status = 409
    exit_code = 8


class ComputeError(MizuotoError):
    """No usable device, or it ran out of memory. ``detail["reason"]`` says which."""

    code = codes.COMPUTE_UNAVAILABLE
    http_status = 503
    exit_code = 9


class WorkerError(MizuotoError):
    """A worker process died or stopped answering. ``detail`` has the exit code and its last lines."""

    code = codes.WORKER_FAILED
    http_status = 502
    exit_code = 10


class EngineError(MizuotoError):
    """An engine raised an exception. ``detail`` has the exception and its traceback."""

    code = codes.ENGINE_ERROR
    http_status = 500
    exit_code = 10


class OperationTimeoutError(MizuotoError):
    """An operation took longer than allowed. ``detail["stage"]`` says where."""

    code = codes.TIMEOUT
    http_status = 504
    exit_code = 11


class DownloadError(MizuotoError):
    """A download failed from every source. ``detail`` has the file, the sources tried and the last error."""

    code = codes.DOWNLOAD_FAILED
    http_status = 502
    exit_code = 12


class InstallError(MizuotoError):
    """Building an engine's environment failed. ``detail["step"]`` names the step, ``detail["tail"]`` its last lines."""

    code = codes.INSTALL_FAILED
    http_status = 500
    exit_code = 13


class RemoteError(MizuotoError):
    """A remote host (``mizuoto agent``) cannot be reached, refused the token, or speaks another protocol."""

    code = codes.REMOTE_UNAVAILABLE
    http_status = 502
    exit_code = 14


class CancelledError(MizuotoError):
    """The operation was cancelled."""

    code = codes.CANCELLED
    http_status = 499
    exit_code = 130


ERROR_CLASSES: dict[str, type[MizuotoError]] = {
    cls.code: cls
    for cls in (
        MizuotoError,
        NotFoundError,
        ConflictError,
        BusyError,
        InvalidPathError,
        ValidationError,
        CapabilityError,
        ParameterError,
        ReferenceInvalidError,
        LanguageError,
        LicenceError,
        ModelMissingError,
        EnvMissingError,
        AssetMissingError,
        DeviceError,
        ComputeError,
        WorkerError,
        EngineError,
        OperationTimeoutError,
        DownloadError,
        InstallError,
        RemoteError,
        CancelledError,
    )
}
ERROR_CLASSES[codes.PARAMETER_OUT_OF_RANGE] = ParameterError


def error_from_code(code: str, message: str, detail: dict[str, Any] | None = None) -> MizuotoError:
    """Rebuild a domain error from a code sent by a worker or stored with a job."""
    cls = ERROR_CLASSES.get(code, MizuotoError)
    if cls is ParameterError:
        err: MizuotoError = ParameterError(message, list((detail or {}).get("fields", [])), detail, out_of_range=code == codes.PARAMETER_OUT_OF_RANGE)
    elif cls is AssetMissingError:
        err = AssetMissingError(message, list((detail or {}).get("assets", [])), detail)
    else:
        err = cls(message, detail)
    if cls is MizuotoError and code != codes.INTERNAL_ERROR:
        err.code = code
    return err
