from __future__ import annotations
from dataclasses import dataclass, field
from xdsl.utils.exceptions import VerifyException
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

if TYPE_CHECKING:
    from xdsl.ir import Operation, Region
    from xdsl.dialects.builtin import StringAttr


@dataclass(frozen=True)
class OpTrait:
    """
    A trait attached to an operation definition.
    Traits can be used to define operation invariants, additional semantic information,
    or to group operations that have similar properties.
    Traits have parameters, which by default is just the `None` value. Parameters should
    always be comparable and hashable.
    Note that traits are the merge of traits and interfaces in MLIR.
    """

    parameters: Any = field(default=None)

    def verify(self, op: Operation) -> None:
        """Check that the operation satisfies the trait requirements."""
        pass


OpTraitInvT = TypeVar("OpTraitInvT", bound=OpTrait)


class Pure(OpTrait):
    """A trait that signals that an operation has no side effects."""


@dataclass(frozen=True)
class HasParent(OpTrait):
    """Constraint the operation to have a specific parent operation."""

    parameters: tuple[type[Operation], ...]

    def __init__(self, parameters: type[Operation] | tuple[type[Operation], ...]):
        if not isinstance(parameters, tuple):
            parameters = (parameters,)
        if len(parameters) == 0:
            raise ValueError("parameters must not be empty")
        super().__init__(parameters)

    def verify(self, op: Operation) -> None:
        parent = op.parent_op()
        if isinstance(parent, tuple(self.parameters)):
            return
        if len(self.parameters) == 1:
            raise VerifyException(
                f"'{op.name}' expects parent op '{self.parameters[0].name}'"
            )
        names = ", ".join([f"'{p.name}'" for p in self.parameters])
        raise VerifyException(f"'{op.name}' expects parent op to be one of {names}")


class IsTerminator(OpTrait):
    """
    This trait provides verification and functionality for operations that are
    known to be terminators.

    https://mlir.llvm.org/docs/Traits/#terminator
    """

    def verify(self, op: Operation) -> None:
        """Check that the operation satisfies the IsTerminator trait requirements."""
        if op.parent is not None and op.parent.last_op != op:
            raise VerifyException(
                f"'{op.name}' must be the last operation in its parent block"
            )


class NoTerminator(OpTrait):
    """
    Allow an operation to have single block regions with no terminator.

    https://mlir.llvm.org/docs/Traits/#terminator
    """

    def verify(self, op: Operation) -> None:
        for region in op.regions:
            if len(region.blocks) > 1:
                raise VerifyException(
                    f"'{op.name}' does not contain single-block regions"
                )


class IsolatedFromAbove(OpTrait):
    """
    Constrains the contained operations to use only values defined inside this
    operation.

    This should be fully compatible with MLIR's Trait:
    https://mlir.llvm.org/docs/Traits/#isolatedfromabove
    """

    def verify(self, op: Operation) -> None:
        # Start by checking all the passed operation's regions
        regions: list[Region] = op.regions.copy()

        # While regions are left to check
        while regions:
            # Pop the first one
            region = regions.pop()
            # Check every block of the region
            for block in region.blocks:
                # Check every operation of the block
                for child_op in block.ops:
                    # Check every operand of the operation
                    for operand in child_op.operands:
                        # The operand must not be defined out of the IsolatedFromAbove op.
                        if not op.is_ancestor(operand.owner):
                            raise VerifyException(
                                "Operation using value defined out of its "
                                "IsolatedFromAbove parent!"
                            )
                    # Check nested regions too; unless the operation is IsolatedFromAbove
                    # too; in which case it will check itself.
                    if not child_op.has_trait(IsolatedFromAbove):
                        regions += child_op.regions


class SymbolOpInterface(OpTrait):
    """
    A `Symbol` is a named operation that resides immediately within a region that defines
    a `SymbolTable` (TODO). A Symbol operation should use the SymbolOpInterface interface to
    provide the necessary verification and accessors.

    Currently the only requirement is a "sym_name" attribute of type StringAttr.

    Please see MLIR documentation for Symbol and SymbolTable for the requirements that are
    upcoming in xDSL.

    https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol
    """

    @staticmethod
    def get_sym_attr_name(op: Operation) -> StringAttr:
        """
        Returns the symbol of the operation
        """
        # import builtin here to avoid circular import
        from xdsl.dialects.builtin import StringAttr

        attr = op.attributes["sym_name"]
        assert isinstance(attr, StringAttr)
        return attr

    def verify(self, op: Operation) -> None:
        # import builtin here to avoid circular import
        from xdsl.dialects.builtin import StringAttr

        if "sym_name" not in op.attributes or not isinstance(
            op.attributes["sym_name"], StringAttr
        ):
            raise VerifyException(
                f'Operation {op.name} must have a "sym_name" attribute of type '
                f"`StringAttr` to conform to {SymbolOpInterface.__name__}"
            )
