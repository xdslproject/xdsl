"""
Core datastructures and solver for dataflow analyses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class ChangeResult(Enum):
    """
    A result type used to indicate if a change happened.
    """

    NO_CHANGE = 0
    CHANGE = 1

    def __or__(self, other: ChangeResult) -> ChangeResult:
        return ChangeResult.CHANGE if self == ChangeResult.CHANGE else other


@dataclass(frozen=True)
class GenericLatticeAnchor(ABC):
    """
    Abstract base class for custom lattice anchors. In dataflow analysis,
    lattices are attached to 'anchors'. These are typically IR constructs
    like SSAValue or ProgramPoint, but can be custom constructs for concepts
    like control-flow edges.
    """

    @abstractmethod
    def __str__(self) -> str: ...
