"""Shared building blocks for native engines.

Each block is written once, imports only ``mizuoto.runtime`` (and torch, numpy), keeps upstream's
state-dict key names, and is tested against the reference implementation it replaces.
"""
