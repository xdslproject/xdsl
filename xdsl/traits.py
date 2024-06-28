from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from xdsl.utils.exceptions import VerifyException

if TYPE_CHECKING:
    from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
    from xdsl.ir import Attribute, Operation, Region
    from xdsl.pattern_rewriter import RewritePattern


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


class ConstantLike(OpTrait):
    """
    Operation known to be constant-like.

    https://mlir.llvm.org/doxygen/classmlir_1_1OpTrait_1_1ConstantLike.html
    """


@dataclass(frozen=True)
class HasParent(OpTrait):
    """Constraint the operation to have a specific parent operation."""

    parameters: tuple[type[Operation], ...]

    def __init__(self, *parameters: type[Operation]):
        if not parameters:
            raise ValueError("parameters must not be empty")
        super().__init__(parameters)

    def verify(self, op: Operation) -> None:
        parent = op.parent_op()
        if isinstance(parent, self.parameters):
            return
        if len(self.parameters) == 1:
            raise VerifyException(
                f"'{op.name}' expects parent op '{self.parameters[0].name}'"
            )
        names = ", ".join(f"'{p.name}'" for p in self.parameters)
        raise VerifyException(f"'{op.name}' expects parent op to be one of {names}")


@dataclass(frozen=True)
class HasAncestor(OpTrait):
    """
    Constraint the operation to have a specific operation as ancestor, i.e. transitive
    parent.
    """

    parameters: tuple[type[Operation], ...]

    def __init__(self, head_param: type[Operation], *tail_params: type[Operation]):
        super().__init__((head_param, *tail_params))

    def verify(self, op: Operation) -> None:
        if self.get_ancestor(op) is None:
            if len(self.parameters) == 1:
                raise VerifyException(
                    f"'{op.name}' expects ancestor op '{self.parameters[0].name}'"
                )
            names = ", ".join(f"'{p.name}'" for p in self.parameters)
            raise VerifyException(
                f"'{op.name}' expects ancestor op to be one of {names}"
            )

    def walk_ancestors(self, op: Operation) -> Iterator[Operation]:
        """Iterates over the ancestors of an operation, including the input"""
        curr = op
        yield curr
        while (curr := curr.parent_op()) is not None:
            yield curr

    def get_ancestor(self, op: Operation) -> Operation | None:
        ancestors = self.walk_ancestors(op)
        matching_ancestors = (a for a in ancestors if isinstance(a, self.parameters))
        return next(matching_ancestors, None)


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
        if not region.blocks:
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
        regions: list[Region] = list(op.regions)

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
                                f"IsolatedFromAbove parent: {child_op}"
                            )
                    # Check nested regions too; unless the operation is IsolatedFromAbove
                    # too; in which case it will check itself.
                    if not child_op.has_trait(IsolatedFromAbove):
                        regions += child_op.regions


class SymbolTable(OpTrait):
    """
    SymbolTable operations are containers for Symbol operations. They offer lookup
    functionality for Symbols, and enforce unique symbols amongst its children.

    A SymbolTable operation is constrained to have a single single-block region.
    """

    def verify(self, op: Operation):
        # import builtin here to avoid circular import
        from xdsl.dialects.builtin import StringAttr

        if len(op.regions) != 1:
            raise VerifyException(
                "Operations with a 'SymbolTable' must have exactly one region"
            )
        if len(op.regions[0].blocks) != 1:
            raise VerifyException(
                "Operations with a 'SymbolTable' must have exactly one block"
            )
        block = op.regions[0].blocks[0]
        met_names: set[StringAttr] = set()
        for o in block.ops:
            if (sym_name := o.get_attr_or_prop("sym_name")) is None:
                continue
            if not isinstance(sym_name, StringAttr):
                continue
            if sym_name in met_names:
                raise VerifyException(f'Redefinition of symbol "{sym_name.data}"')
            met_names.add(sym_name)

    @staticmethod
    def lookup_symbol(
        op: Operation, name: str | StringAttr | SymbolRefAttr
    ) -> Operation | None:
        """
        Lookup a symbol by reference, starting from a specific operation's closest
        SymbolTable parent.
        """
        # import builtin here to avoid circular import
        from xdsl.dialects.builtin import StringAttr, SymbolRefAttr

        anchor: Operation | None = op
        while anchor is not None and not anchor.has_trait(SymbolTable):
            anchor = anchor.parent_op()
        if anchor is None:
            raise ValueError(f"Operation {op} has no SymbolTable ancestor")
        if isinstance(name, str | StringAttr):
            name = SymbolRefAttr(name)
        for o in anchor.regions[0].block.ops:
            if (
                sym_interface := o.get_trait(SymbolOpInterface)
            ) is not None and sym_interface.get_sym_attr_name(o) == name.root_reference:
                if not name.nested_references:
                    return o
                nested_root, *nested_references = name.nested_references.data
                nested_name = SymbolRefAttr(nested_root, nested_references)
                return SymbolTable.lookup_symbol(o, nested_name)
        return None

    @staticmethod
    def insert_or_update(
        symbol_table_op: Operation, symbol_op: Operation
    ) -> Operation | None:
        """
        This takes a symbol_table_op and a symbol_op. It looks if another operation
        inside symbol_table_op already defines symbol_ops symbol. If another operation
        is found, it replaces that operation with symbol_op. Otherwise, symbol_op is
        inserted at the end of symbol_table_op.

        This method returns the operation that was replaced or None if no operation
        was replaced.
        """
        trait = symbol_op.get_trait(SymbolOpInterface)

        if trait is None:
            raise ValueError(
                "Passed symbol_op does not have the SymbolOpInterface trait"
            )

        symbol_name = trait.get_sym_attr_name(symbol_op)

        if symbol_name is None:
            raise ValueError("Passed symbol_op does not have a symbol attribute name")

        tbl_trait = symbol_table_op.get_trait(SymbolTable)

        if tbl_trait is None:
            raise ValueError("Passed symbol_table_op does not have a SymbolTable trait")

        defined_symbol = tbl_trait.lookup_symbol(symbol_table_op, symbol_name)

        if defined_symbol is None:
            symbol_table_op.regions[0].blocks[0].add_op(symbol_op)
            return None
        else:
            parent = defined_symbol.parent
            assert parent is not None
            parent.insert_op_after(symbol_op, defined_symbol)
            parent.detach_op(defined_symbol)
            return defined_symbol


class SymbolOpInterface(OpTrait):
    """
    A `Symbol` is a named operation that resides immediately within a region that defines
    a `SymbolTable` (TODO). A Symbol operation should use the SymbolOpInterface interface to
    provide the necessary verification and accessors.

    A Symbol operation may be optional or not. If - the default - it is not optional,
    a `sym_name` attribute of type StringAttr is required. If it is optional,
    the attribute is optional too.

    xDSL offers OptionalSymbolOpInterface as an always-optional SymbolOpInterface helper.

    More requirements are defined in MLIR; Please see MLIR documentation for Symbol and
    SymbolTable for the requirements that are upcoming in xDSL.

    https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol
    """

    def get_sym_attr_name(self, op: Operation) -> StringAttr | None:
        """
        Returns the symbol of the operation, if any
        """
        # import builtin here to avoid circular import
        from xdsl.dialects.builtin import StringAttr

        sym_name = op.get_attr_or_prop("sym_name")
        if sym_name is None and self.is_optional_symbol(op):
            return None
        if not isinstance(sym_name, StringAttr):
            raise VerifyException(
                f'Operation {op.name} must have a "sym_name" attribute of type '
                f"`StringAttr` to conform to {SymbolOpInterface.__name__}"
            )
        return sym_name

    def is_optional_symbol(self, op: Operation) -> bool:
        """
        Returns true if this operation optionally defines a symbol based on the
        presence of the symbol name.
        """
        return False

    def verify(self, op: Operation) -> None:
        # This helper has the same behaviour, so we reuse it as a verifier.That is, it
        # raises a VerifyException iff this operation is a non-optional symbol *and*
        # there is no "sym_name" attribute or property.
        self.get_sym_attr_name(op)


class OptionalSymbolOpInterface(SymbolOpInterface):
    """
    Helper interface specialization for an optional SymbolOpInterface.
    """

    def is_optional_symbol(self, op: Operation) -> bool:
        return True


class CallableOpInterface(OpTrait, abc.ABC):
    """
    Interface for function-like Operations that can be called in a generic way.

    Please see MLIR documentation for CallOpInterface and CallableOpInterface for more
    information.

    https://mlir.llvm.org/docs/Interfaces/#callinterfaces
    """

    @classmethod
    @abc.abstractmethod
    def get_callable_region(cls, op: Operation) -> Region:
        """
        Returns the body of the operation
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def get_argument_types(cls, op: Operation) -> tuple[Attribute, ...]:
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def get_result_types(cls, op: Operation) -> tuple[Attribute, ...]:
        raise NotImplementedError()


@dataclass(frozen=True)
class HasCanonicalisationPatternsTrait(OpTrait):
    """
    Provides the rewrite passes to canonicalize an operation.

    Each rewrite pattern must have the trait's op as root.
    """

    def verify(self, op: Operation) -> None:
        return

    @classmethod
    @abc.abstractmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        raise NotImplementedError()


class MemoryEffect(OpTrait):
    """
    A trait that enables operations to expose their side-effects or absence thereof.

    NB: The MLIR implementation further allows to describe what *kind* of side-effects
    an operation has, e.g., read-only, or allocation.
    This one is a stripped down version for now, just saying if there are any
    side-effects or not.
    """

    @classmethod
    @abc.abstractmethod
    def has_effects(cls, op: Operation) -> bool:
        raise NotImplementedError()


def is_side_effect_free(op: Operation):
    """
    Boilerplate helper to check if a generic operation is side effect free for sure.
    """
    # If it doesn't say, safely assume it has side effects.
    if not (trait := op.get_trait(MemoryEffect)):
        return False
    return not trait.has_effects(op)


class NoMemoryEffect(MemoryEffect):
    """
    A trait that signals that an operation never has side effects.
    """

    @classmethod
    def has_effects(cls, op: Operation) -> bool:
        return False


class RecursiveMemoryEffect(MemoryEffect):
    """
    A trait that signals that an operation has the side effects of its contained
    operations.

    NB: Upstream, this a separate class, but in our current binary side effect
    implementation, it's easier to have it this way in my opinion.
    """

    @classmethod
    def has_effects(cls, op: Operation) -> bool:
        if not op.regions:
            return True
        return not all(is_side_effect_free(o) for r in op.regions for o in r.walk())


class Pure(NoMemoryEffect):
    """A trait that signals that an operation has no side effects."""


class HasInsnRepresentation(OpTrait, abc.ABC):
    """
    A trait providing information on how to encode an operation using a .insn assember directive.

    The returned string contains python string.format placeholders where formatted operands are inserted during
    printing.

    See https://sourceware.org/binutils/docs/as/RISC_002dV_002dDirectives.html for more information.
    """

    @abc.abstractmethod
    def get_insn(self, op: Operation) -> str:
        """
        Return the insn representation of the operation for printing.
        """
        raise NotImplementedError()
