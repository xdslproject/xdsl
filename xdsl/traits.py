from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from xdsl.utils.exceptions import VerifyException

if TYPE_CHECKING:
    from xdsl.dialects.builtin import StringAttr
    from xdsl.ir import Operation, Region


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


class SingleBlockImplicitTerminator(OpTrait):
    """
    Checks the existence of the specified terminator to an operation which has
    single-block regions.
    The conditions for the implicit creation of the terminator depend on the operation
    and occur during its creation using the `ensure_terminator` method.

    This should be fully compatible with MLIR's Trait:
    https://mlir.llvm.org/docs/Traits/#single-block-with-implicit-terminator
    """

    parameters: type[Operation]

    def verify(self, op: Operation) -> None:
        for region in op.regions:
            if len(region.blocks) > 1:
                raise VerifyException(
                    f"'{op.name}' does not contain single-block regions"
                )
            for block in region.blocks:
                if (last_op := block.last_op) is None:
                    raise VerifyException(
                        f"'{op.name}' contains empty block instead of at least "
                        f"terminating with {self.parameters.name}"
                    )

                if not isinstance(last_op, self.parameters):
                    raise VerifyException(
                        f"'{op.name}' terminates with operation {last_op.name} "
                        f"instead of {self.parameters.name}"
                    )


def ensure_terminator(op: Operation, trait: SingleBlockImplicitTerminator) -> None:
    """
    Method that helps with the creation of an implicit terminator.
    This should be explicitly called during the creation of an operation that has the
    SingleBlockImplicitTerminator trait.
    """

    for region in op.regions:
        if len(region.blocks) > 1:
            raise VerifyException(f"'{op.name}' does not contain single-block regions")

        for block in region.blocks:
            if (
                (last_op := block.last_op) is not None
                and last_op.has_trait(IsTerminator)
                and not isinstance(last_op, trait.parameters)
            ):
                raise VerifyException(
                    f"'{op.name}' terminates with operation {last_op.name} "
                    f"instead of {trait.parameters.name}"
                )

    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block

    for region in op.regions:
        if len(region.blocks) == 0:
            region.add_block(Block())

        for block in region.blocks:
            if (last_op := block.last_op) is None or not last_op.has_trait(
                IsTerminator
            ):
                with ImplicitBuilder(block):
                    trait.parameters.create()


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


class CallableOpInterface(OpTrait, abc.ABC):
    """
    Interface for function-like Operations that can be called in a generic way.

    Please see MLIR documentation for CallOpInterface and CallableOpInterface for more
    information.

    https://mlir.llvm.org/docs/Interfaces/#callinterfaces
    """

    @classmethod
    def get_callable_region(cls, op: Operation) -> Region:
        """
        Returns the body of the operation
        """
        raise NotImplementedError
