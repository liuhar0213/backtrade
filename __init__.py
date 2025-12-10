"""
Top-level package initializer for `backtrade`.

This file intentionally keeps imports minimal. Many core modules live
under the `core/` subpackage; to allow importing them as
`backtrade.optimizer_c2f` (and similar) during test collection and
runtime, we add the `core/` directory to the package search path.

Avoid importing heavy modules here to prevent import-time errors when
test runners (like pytest) import the package for collection.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "RL Trader Team"

import os

# Prefer modules under the `core/` subpackage to be discoverable as
# `backtrade.<module>` by adding the core folder to the package `__path__`.
_core_path = os.path.join(os.path.dirname(__file__), "core")
if os.path.isdir(_core_path) and _core_path not in __path__:
    __path__.insert(0, _core_path)

# Keep the top-level namespace small — users can import from
# `backtrade` or `backtrade.core` as needed.
