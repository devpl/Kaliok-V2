"""Compatibility import for the execution identity contract."""

from kaliok.execution import (
    ExecutionContext,
    ExecutionEnvironment,
    apply_execution_context,
)

__all__ = [
    "ExecutionContext",
    "ExecutionEnvironment",
    "apply_execution_context",
]
