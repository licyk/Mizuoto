"""After a CUDA graph capture fails, does eager work still run, and which cleanup makes it run?

Each way runs in a fresh process: a capture like mizuoto.runtime.graphs.Graphs.capture whose step reads a value back
(.item(), forbidden while capturing), then the cleanup, then plain eager work on the current stream. Run it on an NVIDIA
GPU when a torch release changes how failed captures end (the fallback tests, ``test_a_failed_capture_*``, fail then):

    python scripts/graph_failure_probe.py
"""

import subprocess
import sys

WAYS = ["none", "reset", "synchronize", "end-if-capturing", "end-if-capturing+reset+synchronize", "take-pending-error"]
"""``take-pending-error``: Mizuoto's cleanup (``mizuoto.runtime.graphs.take_pending_error``), what ``Graphs.capture`` does."""

CHILD = r"""
import sys, torch
way = sys.argv[1]
device = torch.device("cuda")
x = torch.ones(4, device=device)
stream = torch.cuda.Stream(device=device)
generator = torch.Generator(device=device)
current = torch.cuda.current_stream(device)
stream.wait_stream(current)
with torch.cuda.stream(stream):
    x.add_(1).sum().item()  # the warm-up: reading back is fine outside a capture
current.wait_stream(stream)
graph = torch.cuda.CUDAGraph()
error = None
try:
    with torch.cuda.stream(stream):
        graph.capture_begin(capture_error_mode="thread_local")
        try:
            x.add_(1).sum().item()
        finally:
            graph.capture_end()
except Exception as e:
    error = f"{type(e).__name__}: {str(e).splitlines()[0]}"
with torch.cuda.stream(stream):
    capturing = torch.cuda.is_current_stream_capturing()
cleanup = []
if "end-if-capturing" in way and capturing:
    with torch.cuda.stream(stream):
        try:
            graph.capture_end()
            cleanup.append("ended")
        except Exception as e:
            cleanup.append(f"end failed: {type(e).__name__}")
if "reset" in way:
    try:
        graph.reset()
        cleanup.append("reset")
    except Exception as e:
        cleanup.append(f"reset failed: {type(e).__name__}")
if way == "take-pending-error":
    from mizuoto.runtime.graphs import take_pending_error

    take_pending_error(device)
    cleanup.append("took the pending error")
if "synchronize" in way:
    try:
        torch.cuda.synchronize(device)
        cleanup.append("synchronized")
    except Exception as e:
        cleanup.append(f"synchronize failed: {type(e).__name__}: {str(e).splitlines()[0]}")
results = []
for attempt in range(3):
    try:
        y = (torch.arange(8, device=device) + torch.tensor([1], device=device)).sum().item()
        results.append(f"ok({y})")
    except Exception as e:
        results.append(f"{type(e).__name__}: {str(e).splitlines()[0]}")
print(f"{way:38} capture error: {error}\n{'':38} still capturing after: {capturing}; cleanup: {cleanup}\n{'':38} eager after, 3 tries: {results}")
"""

if __name__ == "__main__":
    import torch
    import torch.version

    print("torch", torch.__version__, "CUDA", torch.version.cuda, "on", torch.cuda.get_device_name() if torch.cuda.is_available() else "no GPU")
    print("allocator", torch.cuda.get_allocator_backend() if torch.cuda.is_available() else "-")
    for way in WAYS:
        done = subprocess.run([sys.executable, "-c", CHILD, way], capture_output=True, text=True)
        print(done.stdout.rstrip() or done.stderr.strip().splitlines()[-1])
