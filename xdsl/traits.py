from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from typing_extensions import TypeVar

from xdsl.utils.exceptions import VerifyException

if TYPE_CHECKING:
    from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
    from xdsl.ir import Attribute, Operation, Region, SSAValue
    from xdsl.pattern_rewriter import RewritePattern


@dataclass(frozen=True)
class OpTrait:
    """
    A trait attached to an operation definition.
    Traits can be used to define operation invariants, additional semantic information,
    or to group operations that have similar properties.
    Note that traits are the merge of traits and interfaces in MLIR.
    """

    def verify(self, op: Operation) -> None:
        """Check that the operation satisfies the trait requirements."""
        pass


OpTraitInvT = TypeVar("OpTraitInvT", bound=OpTrait)


class ConstantLike(OpTrait, abc.ABC):
    """
    Operation known to be constant-like.

    See external [documentation](https://mlir.llvm.org/doxygen/classmlir_1_1OpTrait_1_1ConstantLike.html).
    """

    @classmethod
    @abc.abstractmethod
    def get_constant_value(cls, op: Operation) -> Attribute:
        """
        Get the constant value from this constant-like operation.

        Returns:
            The constant value as an Attribute, or None if the value cannot be determined.
        """
        raise NotImplementedError()


@dataclass(frozen=True)
class HasParent(OpTrait):
    """Constraint the operation to have a specific parent operation."""

    op_types: tuple[type[Operation], ...]

    def __init__(self, head_param: type[Operation], *tail_params: type[Operation]):
        object.__setattr__(self, "op_types", (head_param, *tail_params))

    def verify(self, op: Operation) -> None:
        parent = op.parent_op()
        # Don't check parent when op is detached
        if parent is None:
            return
        if isinstance(parent, self.op_types):
            return
        if len(self.op_types) == 1:
            raise VerifyException(
                f"'{op.name}' expects parent op '{self.op_types[0].name}'"
            )
        names = ", ".join(f"'{p.name}'" for p in self.op_types)
        raise VerifyException(f"'{op.name}' expects parent op to be one of {names}")


@dataclass(frozen=True)
class HasAncestor(OpTrait):
    """
    Constraint the operation to have a specific operation as ancestor, i.e. transitive
    parent.
    """

    op_types: tuple[type[Operation], ...]

    def __init__(self, head_param: type[Operation], *tail_params: type[Operation]):
        object.__setattr__(self, "op_types", (head_param, *tail_params))

    def verify(self, op: Operation) -> None:
        if self.get_ancestor(op) is None:
            if len(self.op_types) == 1:
                raise VerifyException(
                    f"'{op.name}' expects ancestor op '{self.op_types[0].name}'"
                )
            names = ", ".join(f"'{p.name}'" for p in self.op_types)
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
        matching_ancestors = (a for a in ancestors if isinstance(a, self.op_types))
        return next(matching_ancestors, None)


class IsTerminator(OpTrait):
    """
    This trait provides verification and functionality for operations that are
    known to be terminators.

    See external [documentation](https://mlir.llvm.org/docs/Traits/#terminator).
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

    See external [documentation](https://mlir.llvm.org/docs/Traits/#terminator).
    """

    def verify(self, op: Operation) -> None:
        for region in op.regions:
            if len(region.blocks) > 1:
                raise VerifyException(
                    f"'{op.name}' does not contain single-block regions"
                )


@dataclass(frozen=True)
class SingleBlockImplicitTerminator(OpTrait):
    """
    Checks the existence of the specified terminator to an operation which has
    single-block regions.
    The conditions for the implicit creation of the terminator depend on the operation
    and occur during its creation using the `ensure_terminator` method.

    This should be fully compatible with MLIR's Trait.

    See external [documentation](https://mlir.llvm.org/docs/Traits/#single-block-with-implicit-terminator).
    """

    op_type: type[Operation]

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
                        f"terminating with {self.op_type.name}"
                    )

                if not isinstance(last_op, self.op_type):
                    raise VerifyException(
                        f"'{op.name}' terminates with operation {last_op.name} "
                        f"instead of {self.op_type.name}"
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

        from xdsl.dialects.builtin import UnregisteredOp

        for block in region.blocks:
            if (
                (last_op := block.last_op) is not None
                and not isinstance(last_op, UnregisteredOp)
                and last_op.has_trait(IsTerminator)
                and not isinstance(last_op, trait.op_type)
            ):
                raise VerifyException(
                    f"'{op.name}' terminates with operation {last_op.name} "
                    f"instead of {trait.op_type.name}"
                )

    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block

    for region in op.regions:
        if not region.blocks:
            region.add_block(Block())

        for block in region.blocks:
            if (last_op := block.last_op) is None or not last_op.has_trait(
                IsTerminator, value_if_unregistered=False
            ):
                with ImplicitBuilder(block):
                    trait.op_type.create()


class IsolatedFromAbove(OpTrait):
    """
    Constrains the contained operations to use only values defined inside this
    operation.

    This should be fully compatible with MLIR's Trait.

    See external [documentation](https://mlir.llvm.org/docs/Traits/#isolatedfromabove).
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


class SymbolUserOpInterface(OpTrait, abc.ABC):
    """
    Used to represent operations that reference Symbol operations. This provides the
    ability to perform safe and efficient verification of symbol uses, as well as
    additional functionality.

    See external [documentation](https://mlir.llvm.org/docs/Interfaces/#symbolinterfaces).
    """

    @abc.abstractmethod
    def verify(self, op: Operation) -> None:
        """
        This method should be adapted to the requirements of specific symbol users per
        operation.

        It corresponds to the verifySymbolUses in upstream MLIR.
        """
        raise NotImplementedError()


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

    See external [documentation](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol).
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

    See external [documentation](https://mlir.llvm.org/docs/Interfaces/#callinterfaces).
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
class HasCanonicalizationPatternsTrait(OpTrait):
    """
    Provides the rewrite passes to canonicalize an operation.

    Each rewrite pattern must have the trait's op as root.
    """

    def get_patterns(
        self,
        op: type[Operation],
    ) -> tuple[RewritePattern, ...]:
        return type(self).get_canonicalization_patterns()

    @classmethod
    @abc.abstractmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        raise NotImplementedError()


@dataclass(frozen=True)
class HasShapeInferencePatternsTrait(OpTrait):
    """
    Provides the rewrite passes to shape infer an operation.

    Each rewrite pattern must have the trait's op as root.
    """

    def verify(self, op: Operation) -> None:
        return

    @classmethod
    @abc.abstractmethod
    def get_shape_inference_patterns(cls) -> tuple[RewritePattern, ...]:
        raise NotImplementedError()


class MemoryEffectKind(Enum):
    """
    The kind of side effect an operation can have.

    MLIR has a more detailed version of this, able to tie effects to specfic resources or
    values. Here, everything has its effect on the universe.
    """

    READ = auto()
    """
    Indicates that the operation reads from some resource. A 'read' effect implies only
    dereferencing of the resource, and not any visible mutation.
    """

    WRITE = auto()
    """
    Indicates that the operation writes to some resource. A 'write' effect implies only
    mutating a resource, and not any visible dereference or read.
    """

    ALLOC = auto()
    """
    Indicates that the operation allocates from some resource. An 'allocate' effect
    implies only allocation of the resource, and not any visible mutation or dereference.
    """

    FREE = auto()
    """
    Indicates that the operation frees some resource that has been allocated. A 'free'
    effect implies only de-allocation of the resource, and not any visible allocation,
    mutation or dereference.
    """


@dataclass(frozen=True)
class EffectInstance:
    """
    An instance of a side effect.
    """

    kind: MemoryEffectKind
    """
    The kind of side effect.
    """

    value: SSAValue | SymbolRefAttr | None = field(default=None)
    """
    The value or symbol that is affected by the side effect, if known.
    """


class MemoryEffect(OpTrait):
    """
    A trait that enables operations to expose their side-effects or absence thereof.
    """

    @classmethod
    @abc.abstractmethod
    def get_effects(cls, op: Operation) -> set[EffectInstance] | None:
        """
        Returns the concrete side effects of the operation.

        Return None if the operation cannot conclude - interpreted as if the operation
        had no MemoryEffect interface in the first place.
        """
        raise NotImplementedError()


def has_effects(op: Operation, effect: MemoryEffectKind) -> bool:
    """
    Returns if the operation has side effects of this kind.
    """
    effects = get_effects(op)
    return effects is not None and any(e.kind == effect for e in effects)


def has_exact_effect(op: Operation, effect: MemoryEffectKind) -> bool:
    """
    Returns if the operation has the given side effects and no others.

    proxy for only_has_effect
    """
    return only_has_effect(op, effect)


def only_has_effect(op: Operation, effect: MemoryEffectKind) -> bool:
    """
    Returns if the operation has the given side effects and no others.
    """
    effects = get_effects(op)
    return effects is not None and all(e.kind == effect for e in effects)


def is_side_effect_free(op: Operation) -> bool:
    """
    Boilerplate helper to check if a generic operation is side effect free for sure.
    """
    effects = get_effects(op)
    return effects is not None and len(effects) == 0


def get_effects(op: Operation) -> set[EffectInstance] | None:
    """
    Helper to get known side effects of an operation.
    None means that the operation has unknown effects, for safety.
    """

    effect_interfaces = op.get_traits_of_type(MemoryEffect)
    if not effect_interfaces:
        return None

    effects = set[EffectInstance]()
    for it in op.get_traits_of_type(MemoryEffect):
        it_effects = it.get_effects(op)
        if it_effects is None:
            return None
        effects.update(it_effects)

    return effects


class NoMemoryEffect(MemoryEffect):
    """
    A trait that signals that an operation never has side effects.
    """

    @classmethod
    def get_effects(cls, op: Operation) -> set[EffectInstance]:
        return set()


class MemoryReadEffect(MemoryEffect):
    """
    A trait that signals that an operation always has read side effects.
    """

    @classmethod
    def get_effects(cls, op: Operation) -> set[EffectInstance]:
        return {EffectInstance(MemoryEffectKind.READ)}


class MemoryWriteEffect(MemoryEffect):
    """
    A trait that signals that an operation always has write side effects.
    """

    @classmethod
    def get_effects(cls, op: Operation) -> set[EffectInstance]:
        return {EffectInstance(MemoryEffectKind.WRITE)}


class MemoryAllocEffect(MemoryEffect):
    """
    A trait that signals that an operation always has alloc side effects.
    """

    @classmethod
    def get_effects(cls, op: Operation) -> set[EffectInstance]:
        return {EffectInstance(MemoryEffectKind.ALLOC)}


class MemoryFreeEffect(MemoryEffect):
    """
    A trait that signals that an operation always has deallocation side effects.
    """

    @classmethod
    def get_effects(cls, op: Operation) -> set[EffectInstance]:
        return {EffectInstance(MemoryEffectKind.FREE)}


class RecursiveMemoryEffect(MemoryEffect):
    """
    A trait that signals that an operation has the side effects of its contained
    operations.
    """

    @classmethod
    def get_effects(cls, op: Operation):
        effects = set[EffectInstance]()
        for r in op.regions:
            for b in r.blocks:
                for child_op in b.ops:
                    child_effects = get_effects(child_op)
                    if child_effects is None:
                        return None
                    effects.update(child_effects)
        return effects


class ConditionallySpeculatable(OpTrait):
    @classmethod
    @abc.abstractmethod
    def is_speculatable(cls, op: Operation) -> bool:
        raise NotImplementedError()


class AlwaysSpeculatable(ConditionallySpeculatable):
    @classmethod
    def is_speculatable(cls, op: Operation):
        return True


class RecursivelySpeculatable(ConditionallySpeculatable):
    @classmethod
    def is_speculatable(cls, op: Operation):
        return all(
            is_speculatable(o) for r in op.regions for b in r.blocks for o in b.ops
        )


def is_speculatable(op: Operation):
    trait = op.get_trait(ConditionallySpeculatable)
    return (trait is not None) and trait.is_speculatable(op)


class Pure(NoMemoryEffect, AlwaysSpeculatable):
    """
    In MLIR, Pure is NoMemoryEffect + AlwaysSpeculatable, but the latter is nowhere to be
    found here.
    """


class Commutative(OpTrait):
    """
    A trait that signals that an operation is commutative.
    """


class HasInsnRepresentation(OpTrait, abc.ABC):
    """
    A trait providing information on how to encode an operation using a .insn assember directive.

    The returned string contains python string.format placeholders where formatted operands are inserted during
    printing.

    See external [documentation](https://sourceware.org/binutils/docs/as/RISC_002dV_002dDirectives.html for more information.).
    """

    @abc.abstractmethod
    def get_insn(self, op: Operation) -> str:
        """
        Return the insn representation of the operation for printing.
        """
        raise NotImplementedError()


@dataclass(frozen=True)
class SameOperandsAndResultType(OpTrait):
    """Constrain the operation to have the same operands and result type."""

    def verify(self, op: Operation) -> None:
        from xdsl.utils.type import (
            get_element_type_or_self,
            get_encoding,
            have_compatible_shape,
        )

        if len(op.results) < 1 or len(op.operands) < 1:
            raise VerifyException(
                f"'{op.name}' requires at least one result or operand"
            )

        result_type0 = get_element_type_or_self(op.result_types[0])

        encoding = get_encoding(op.result_types[0])

        for result_type in op.result_types[1:]:
            result_type_elem = get_element_type_or_self(result_type)
            if result_type0 != result_type_elem or not have_compatible_shape(
                op.result_types[0], result_type
            ):
                raise VerifyException(
                    f"'{op.name} requires the same type for all operands and results"
                )

            element_encoding = get_encoding(result_type)

            if encoding != element_encoding:
                raise VerifyException(
                    f"'{op.name} requires the same encoding for all operands and results"
                )

        for operand_type in op.operand_types:
            operand_type_elem = get_element_type_or_self(operand_type)
            if result_type0 != operand_type_elem or not have_compatible_shape(
                op.result_types[0], operand_type
            ):
                raise VerifyException(
                    f"'{op.name} requires the same type for all operands and results"
                )

            element_encoding = get_encoding(operand_type)

            if encoding != element_encoding:
                raise VerifyException(
                    f"'{op.name} requires the same encoding for all operands and results"
                )
