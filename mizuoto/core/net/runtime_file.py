"""``server.json`` in the data directory, so other programs can find the running server."""

import json
import os
import sys
from pathlib import Path
from typing import Any

RUNTIME_FILE_NAME = "server.json"


def write_runtime_file(data_dir: Path, host: str, port: int, url: str) -> Path:
    path = data_dir / RUNTIME_FILE_NAME
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"host": host, "port": port, "url": url, "pid": os.getpid()}), encoding="utf-8")
    os.replace(tmp, path)
    return path


def remove_runtime_file(data_dir: Path) -> None:
    path = data_dir / RUNTIME_FILE_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if data.get("pid") == os.getpid():
        path.unlink(missing_ok=True)


def _windows_pid_alive(pid: int) -> bool:
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    process_query_limited_information = 0x1000
    error_access_denied = 5
    still_active = 259
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        # ERROR_INVALID_PARAMETER (87): no such process; access denied: it exists.
        return ctypes.get_last_error() == error_access_denied
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def pid_alive(pid: int) -> bool:
    """Whether a process with this id is running."""
    return _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # os.kill(pid, 0) on Windows is TerminateProcess(pid, 0), not a probe.
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def read_runtime_file(data_dir: Path) -> dict[str, Any] | None:
    """Return the runtime file's contents, or None if it is missing or stale."""
    try:
        data = json.loads((data_dir / RUNTIME_FILE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pid = data.get("pid")
    if not isinstance(pid, int) or not _pid_alive(pid):
        return None
    return data
