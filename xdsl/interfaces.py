"""
Interfaces are a convenience wrapper around traits, providing logic on the operation
directly.

Operations inherit all the traits of their superclasses, allowing them to combine
behaviours via sublcassing.
This can be more convenient than adding the traits explicitly.
"""

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from typing_extensions import override

from xdsl.ir import Attribute, Operation, Region, SSAValue
from xdsl.irdl import traits_def
from xdsl.pattern_rewriter import RewritePattern
from xdsl.traits import (
    ConditionallySpeculatable,
    HasCanonicalizationPatternsTrait,
    HasFolder,
    InvocationBounds,
    OperandRange,
    RegionBranchOpTrait,
    RegionBranchPoint,
    RegionBranchTerminatorOpTrait,
    RegionSuccessor,
)


@dataclass(frozen=True)
class _HasCanonicalizationPatternsInterfaceTrait(HasCanonicalizationPatternsTrait):
    """
    Gets the canonicalization patterns from the operation's implementation
    of `HasCanonicalizationPatternsInterface`.
    """

    def verify(self, op: Operation) -> None:
        return

    def get_patterns(
        self,
        op: type[Operation],
    ) -> tuple[RewritePattern, ...]:
        op = cast(type[HasCanonicalizationPatternsInterface], op)
        return op.get_canonicalization_patterns()


class HasCanonicalizationPatternsInterface(Operation, abc.ABC):
    """
    An operation subclassing this interface must implement the
    `get_canonicalization_patterns` method, which returns a tuple of patterns that
    canonicalize this operation.
    Wraps `CanonicalizationPatternsTrait`.
    """

    traits = traits_def(_HasCanonicalizationPatternsInterfaceTrait())

    @classmethod
    @abc.abstractmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        raise NotImplementedError()


class _HasFolderInterfaceTrait(HasFolder):
    """
    Gets the fold results from the operation's implementation
    of `HasFolderInterface`.
    """

    def verify(self, op: Operation) -> None:
        return

    @classmethod
    def fold(cls, op: Operation):
        op = cast(HasFolderInterface, op)
        return op.fold()


class HasFolderInterface(Operation, abc.ABC):
    """
    An operation subclassing this interface must implement the
    `fold` method, which attempts to fold the operation.
    Wraps `HasFolderTrait`.
    """

    traits = traits_def(_HasFolderInterfaceTrait())

    @abc.abstractmethod
    def fold(self) -> Sequence[SSAValue | Attribute] | None:
        """
        Attempts to fold the operation. The fold method cannot modify the IR.
        Returns either an existing SSAValue or an Attribute for each result of the operation.
        When folding is unsuccessful, returns None.

        The fold method is not allowed to mutate the operation being folded.
        """
        raise NotImplementedError()


@dataclass(frozen=True)
class _ConditionallySpeculatableInterfaceTrait(ConditionallySpeculatable):
    """
    Gets the speculatability from the operation's implementation
    of `ConditionallySpeculatableInterface`.
    """

    def verify(self, op: Operation) -> None:
        return

    @classmethod
    def is_speculatable(cls, op: Operation) -> bool:
        op = cast("ConditionallySpeculatableInterface", op)
        return op.is_speculatable()


class ConditionallySpeculatableInterface(Operation, abc.ABC):
    """
    An operation subclassing this interface must implement the
    `is_speculatable` method, which indicates whether the operation
    can be speculatively executed.
    """

    traits = traits_def(_ConditionallySpeculatableInterfaceTrait())

    @abc.abstractmethod
    def is_speculatable(self) -> bool:
        """
        Returns True if this operation can be speculatively executed,
        False otherwise. Speculatable ops have no undefined behavior
        and no observable side effects.
        """
        raise NotImplementedError()


class _RegionBranchOpInterfaceTrait(RegionBranchOpTrait):
    """
    Gets the region branching behaviour from the operation's implementation of
    `RegionBranchOpInterface`.
    """

    @override
    @classmethod
    def entry_successor_operands(
        cls, op: Operation, successor: RegionSuccessor
    ) -> OperandRange:
        return cast(RegionBranchOpInterface, op).entry_successor_operands(successor)

    @override
    @classmethod
    def entry_successor_regions(
        cls, op: Operation, operands: Sequence[Attribute | None]
    ) -> Sequence[RegionSuccessor]:
        return cast(RegionBranchOpInterface, op).entry_successor_regions(operands)

    @override
    @classmethod
    def successor_regions(
        cls, op: Operation, point: RegionBranchPoint
    ) -> Sequence[RegionSuccessor]:
        return cast(RegionBranchOpInterface, op).successor_regions(point)

    @override
    @classmethod
    def successor_regions_from_region(
        cls, op: Operation, region: Region
    ) -> Sequence[RegionSuccessor]:
        return cast(RegionBranchOpInterface, op).successor_regions_from_region(region)

    @override
    @classmethod
    def successor_inputs(
        cls, op: Operation, successor: RegionSuccessor
    ) -> Sequence[SSAValue]:
        return cast(RegionBranchOpInterface, op).successor_inputs(successor)

    @override
    @classmethod
    def predecessors(
        cls, op: Operation, successor: RegionSuccessor
    ) -> Sequence[RegionBranchPoint]:
        return cast(RegionBranchOpInterface, op).predecessors(successor)

    @override
    @classmethod
    def predecessor_values(
        cls, op: Operation, successor: RegionSuccessor, index: int
    ) -> Sequence[SSAValue]:
        return cast(RegionBranchOpInterface, op).predecessor_values(successor, index)

    @override
    @classmethod
    def region_invocation_bounds(
        cls, op: Operation, operands: Sequence[Attribute | None]
    ) -> Sequence[InvocationBounds]:
        return cast(RegionBranchOpInterface, op).region_invocation_bounds(operands)

    @override
    @classmethod
    def are_types_compatible(
        cls, op: Operation, lhs: Attribute, rhs: Attribute
    ) -> bool:
        return cast(RegionBranchOpInterface, op).are_types_compatible(lhs, rhs)


class RegionBranchOpInterface(Operation, abc.ABC):
    """
    An operation subclassing this interface must implement `successor_regions`, and may
    override the other methods, whose semantics are documented on `RegionBranchOpTrait`.
    Wraps `RegionBranchOpTrait`.
    """

    traits = traits_def(_RegionBranchOpInterfaceTrait())

    def entry_successor_operands(self, successor: RegionSuccessor) -> OperandRange:
        """See `RegionBranchOpTrait.entry_successor_operands`."""
        return OperandRange(self, 0, 0)

    def entry_successor_regions(
        self, operands: Sequence[Attribute | None]
    ) -> Sequence[RegionSuccessor]:
        """See `RegionBranchOpTrait.entry_successor_regions`."""
        return self.successor_regions(RegionBranchPoint.PARENT)

    @abc.abstractmethod
    def successor_regions(self, point: RegionBranchPoint) -> Sequence[RegionSuccessor]:
        """See `RegionBranchOpTrait.successor_regions`."""
        raise NotImplementedError()

    def successor_regions_from_region(
        self, region: Region
    ) -> Sequence[RegionSuccessor]:
        """See `RegionBranchOpTrait.successor_regions_from_region`."""
        regions: list[RegionSuccessor] = []
        for block in region.blocks:
            terminator = block.last_op
            if terminator is not None and terminator.has_trait(
                RegionBranchTerminatorOpTrait
            ):
                regions.extend(self.successor_regions(RegionBranchPoint(terminator)))
        return regions

    def successor_inputs(self, successor: RegionSuccessor) -> Sequence[SSAValue]:
        """See `RegionBranchOpTrait.successor_inputs`."""
        return ()

    def predecessors(self, successor: RegionSuccessor) -> Sequence[RegionBranchPoint]:
        """See `RegionBranchOpTrait.predecessors`."""
        return [
            point
            for point in _RegionBranchOpInterfaceTrait.all_region_branch_points(self)
            if successor in self.successor_regions(point)
        ]

    def predecessor_values(
        self, successor: RegionSuccessor, index: int
    ) -> Sequence[SSAValue]:
        """See `RegionBranchOpTrait.predecessor_values`."""
        trait = _RegionBranchOpInterfaceTrait
        return [
            trait.successor_operands(self, predecessor, successor)[index]
            for predecessor in self.predecessors(successor)
        ]

    def region_invocation_bounds(
        self, operands: Sequence[Attribute | None]
    ) -> Sequence[InvocationBounds]:
        """See `RegionBranchOpTrait.region_invocation_bounds`."""
        return [InvocationBounds.UNKNOWN] * len(self.regions)

    def are_types_compatible(self, lhs: Attribute, rhs: Attribute) -> bool:
        """See `RegionBranchOpTrait.are_types_compatible`."""
        return lhs == rhs


class _RegionBranchTerminatorOpInterfaceTrait(RegionBranchTerminatorOpTrait):
    """
    Gets the region branching behaviour from the operation's implementation of
    `RegionBranchTerminatorOpInterface`.
    """

    @override
    @classmethod
    def successor_operands(
        cls, op: Operation, successor: RegionSuccessor
    ) -> OperandRange:
        return cast(RegionBranchTerminatorOpInterface, op).successor_operands(successor)

    @override
    @classmethod
    def successor_regions(
        cls, op: Operation, operands: Sequence[Attribute | None]
    ) -> Sequence[RegionSuccessor]:
        return cast(RegionBranchTerminatorOpInterface, op).successor_regions(operands)


class RegionBranchTerminatorOpInterface(Operation, abc.ABC):
    """
    An operation subclassing this interface may override its methods, whose semantics are
    documented on `RegionBranchTerminatorOpTrait`. Wraps `RegionBranchTerminatorOpTrait`.
    """

    traits = traits_def(_RegionBranchTerminatorOpInterfaceTrait())

    def successor_operands(self, successor: RegionSuccessor) -> OperandRange:
        """See `RegionBranchTerminatorOpTrait.successor_operands`."""
        return OperandRange(self)

    def successor_regions(
        self, operands: Sequence[Attribute | None]
    ) -> Sequence[RegionSuccessor]:
        """See `RegionBranchTerminatorOpTrait.successor_regions`."""
        return RegionBranchTerminatorOpTrait.successor_regions(self, operands)
