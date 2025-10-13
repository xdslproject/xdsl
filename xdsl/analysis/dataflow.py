"""
Core datastructures and solver for dataflow analyses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from xdsl.ir import Block, Operation


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


@dataclass(frozen=True)
class ProgramPoint:
    """
    A point in the program, either before an operation or at the end of a block.
    This is used as an anchor for dense analyses.
    """

    entity: Operation | Block

    @staticmethod
    def before(op: Operation) -> ProgramPoint:
        """Returns the program point just before an operation."""
        return ProgramPoint(op)

    @staticmethod
    def after(op: Operation) -> ProgramPoint:
        """Returns the program point just after an operation."""
        if op.next_op is not None:
            return ProgramPoint(op.next_op)
        if op.parent is None:
            raise ValueError("Cannot get ProgramPoint after a detached operation.")
        return ProgramPoint(op.parent)

    @staticmethod
    def at_start_of_block(block: Block) -> ProgramPoint:
        """Returns the program point at the start of a block."""
        if block.first_op is None:
            # If block is empty, the start is the end.
            return ProgramPoint(block)
        return ProgramPoint(block.first_op)

    @staticmethod
    def at_end_of_block(block: Block) -> ProgramPoint:
        """Returns the program point at the end of a block."""
        return ProgramPoint(block)

    @property
    def op(self) -> Operation | None:
        """The operation this point is before, or None if at the end of a block."""
        if isinstance(self.entity, Operation):
            return self.entity
        return None

    @property
    def block(self) -> Block | None:
        """The block this point is in, or the block itself if at the end."""
        if isinstance(self.entity, Operation):
            return self.entity.parent
        return self.entity

    def __str__(self) -> str:
        if isinstance(self.entity, Operation):
            return f"before op '{self.entity.name}'"
        return f"at end of block '{self.entity.name_hint or id(self.entity)}'"
