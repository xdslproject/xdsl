"""
Core datastructures and solver for dataflow analyses.
"""

from __future__ import annotations

from enum import Enum


class ChangeResult(Enum):
    """
    A result type used to indicate if a change happened.
    """

    NO_CHANGE = 0
    CHANGE = 1

    def __or__(self, other: ChangeResult) -> ChangeResult:
        return ChangeResult.CHANGE if self == ChangeResult.CHANGE else other
