"""Mizuoto's native runtime: engines in this process.

The framework every in-tree engine is built on:

- ``context``: the request context. Cancel token, per-request generators, progress, streamed text.
- ``engine``: the engine contract and its registry.
- ``device``: devices, memory and dtypes.
- ``handle`` / ``manager``: model handles and the manager that places them on devices.
- ``ops`` / ``loading``: layers and state-dict loading; ``accel``: attention and its backends.
- ``snapshot``: the global-state snapshot tests take around engine calls.

Importing ``mizuoto.runtime`` loads no torch; the modules that need it import it themselves.
"""
