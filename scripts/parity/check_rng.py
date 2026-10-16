"""Check that explicit generators reproduce torch's seeded default stream.

    <python> scripts/parity/check_rng.py [--device cuda:0]

Native engines draw from a ``torch.Generator`` seeded per request. Upstream code draws from the
default generator after ``torch.manual_seed``. For native and upstream draws to match, both streams
must be the same for every random op the engines use. This prints one JSON line per (device, op) and
exits non-zero if any differs. It needs only torch, so it runs in every reference environment and in
the native one.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any


def ops(torch: ModuleType, device: str) -> dict[str, Callable[[Any], Any]]:
    shape = (3, 1000)
    probs = torch.softmax(torch.arange(64, dtype=torch.float32, device=device) / 7.0, dim=0).expand(4, 64).contiguous()
    return {
        "randn": lambda g: torch.randn(shape, device=device, generator=g),
        "randn_fp16": lambda g: torch.randn(shape, device=device, dtype=torch.float16, generator=g),
        "randn_bf16": lambda g: torch.randn(shape, device=device, dtype=torch.bfloat16, generator=g),
        "rand": lambda g: torch.rand(shape, device=device, generator=g),
        "randint": lambda g: torch.randint(0, 1000, shape, device=device, generator=g),
        "multinomial": lambda g: torch.multinomial(probs, 2, generator=g),
        "multinomial_replace": lambda g: torch.multinomial(probs, 3, replacement=True, generator=g),
        "exponential_": lambda g: torch.empty(shape, device=device).exponential_(generator=g),
        "uniform_": lambda g: torch.empty(shape, device=device).uniform_(generator=g),
        "normal_": lambda g: torch.empty(shape, device=device).normal_(generator=g),
        "bernoulli": lambda g: torch.bernoulli(torch.full(shape, 0.3, device=device), generator=g),
        # three ops in a row: the stream must also advance the same way
        "sequence": lambda g: torch.cat(
            [torch.rand(17, device=device, generator=g), torch.randn(5, device=device, generator=g), torch.multinomial(probs[0], 1, generator=g).float()]
        ),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", action="append", help="cpu (always) plus e.g. cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    import torch

    devices = ["cpu", *(d for d in (args.device or []) if d != "cpu")]
    bad = 0
    for device in devices:
        for name, op in ops(torch, device).items():
            # upstream style: global seed, default generator (``None``)
            torch.manual_seed(args.seed)
            ref = op(None)
            # native style: a fresh generator on the device
            g = torch.Generator(device=device).manual_seed(args.seed)
            ours = op(g)
            same = bool(torch.equal(ref, ours))
            bad += not same
            print(json.dumps({"torch": torch.__version__, "device": device, "op": name, "identical": same}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
