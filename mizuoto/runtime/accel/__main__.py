"""``python -m mizuoto.runtime.accel [device ...]``: whether each attention backend runs on each device (default: every
CUDA device), as JSON ``{device: {backend: reason or null}}``, for ``mizuoto doctor`` without torch in the host."""

from __future__ import annotations

import json
import sys


def report(devices: list[str]) -> dict[str, dict[str, str | None]]:
    import torch

    from mizuoto.runtime.accel import backends
    from mizuoto.runtime.accel.registry import REGISTRY

    backends.register_all()
    if not devices:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    return {device: REGISTRY.report(device) for device in devices}


def main() -> None:
    print(json.dumps(report(sys.argv[1:])))


if __name__ == "__main__":
    main()
