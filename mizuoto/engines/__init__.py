"""Mizuoto's native engine implementations, one package per engine.

Each package defines an ``Engine`` subclass named in ``mizuoto.runtime.engine.BUILTIN``. Engines import
``mizuoto.runtime``, ``mizuoto.nn`` and ``mizuoto.text``: never another engine, never ``mizuoto.core``.
"""
