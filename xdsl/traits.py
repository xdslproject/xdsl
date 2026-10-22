from __future__ import annotations

import abc
from collections.abc import Callable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, ClassVar, TypeAlias, overload

from typing_extensions import TypeVar, override

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

    To participate in constant folding and other generic mechanisms implement
    `HasFolder` or `HasFolderInterface` for your operation.

    See external [documentation](https://mlir.llvm.org/doxygen/classmlir_1_1OpTrait_1_1ConstantLike.html).
    """

    @staticmethod
    def get_constant_value(ssa_value: SSAValue) -> Attribute | None:
        """
        If the value is the result of a `ConstantLike` operation that implements
        `HasFolderInterface`, return the attribute returned by `fold` corresponding to
        the value's index in the list of results.
        """
        from xdsl.ir import Attribute, OpResult

        if (
            isinstance(ssa_value, OpResult)
            and (op := ssa_value.owner)
            and op.has_trait(ConstantLike)
            and (t := op.get_trait(HasFolder)) is not None
            and (values := t.fold(op)) is not None
            and isinstance(value := values[ssa_value.index], Attribute)
        ):
            return value


class HasFolder(OpTrait):
    """
    Operation known to support folding.
    """

    @classmethod
    @abc.abstractmethod
    def fold(cls, op: Operation) -> Sequence[SSAValue | Attribute] | None:
        """
        Attempts to fold the operation. The fold method cannot modify the IR.
        Returns either an existing SSAValue or an Attribute for each result of the operation.
        When folding is unsuccessful, returns None.
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


class ReturnLike(OpTrait):
    """
    This trait indicates that a terminator operation is "return-like". This
    means that it exits its current region and forwards its operands as "exit"
    values to the parent region. Operations with this trait are not permitted to
    contain successors or produce results.
    """

    def verify(self, op: Operation) -> None:
        if not op.has_trait(IsTerminator):
            raise VerifyException(f"{op.name} is not a terminator")
        if op.results:
            raise VerifyException(f"{op.name} does not have zero results")
        if op.successors:
            raise VerifyException(f"{op.name} does not have zero successors")


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
                        # Since op.is_ancestor(op) evaluates to True, also check if the operand is
                        # owned by the op itself (not allowed).
                        if not op.is_ancestor(operand.owner) or operand.owner is op:
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

    MLIR has a more detailed version of this, able to tie effects to specific resources or
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
class Resource(abc.ABC):
    """
    This class represents a specific resource that an effect applies to.
    """

    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class DefaultResource(Resource):
    """
    A conservative default resource kind.
    """

    def name(self) -> str:
        return "<Default>"


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

    resource: Resource = field(default=DefaultResource())
    """
    The resource that the effect applies to.
    """


class MemoryEffect(OpTrait):
    """
    A trait that enables operations to expose their side-effects or absence thereof.
    """

    @classmethod
    @abc.abstractmethod
    def get_effects(cls, op: Operation) -> AbstractSet[EffectInstance] | None:
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


def get_effects(op: Operation) -> AbstractSet[EffectInstance] | None:
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
    def get_effects(cls, op: Operation) -> AbstractSet[EffectInstance]:
        return frozenset()


# Frozen sets to avoid re-creating a set every time
_MEMORY_READ_EFFECT_SET = frozenset((EffectInstance(MemoryEffectKind.READ),))
_MEMORY_WRITE_EFFECT_SET = frozenset((EffectInstance(MemoryEffectKind.WRITE),))
_MEMORY_ALLOC_EFFECT_SET = frozenset((EffectInstance(MemoryEffectKind.ALLOC),))
_MEMORY_FREE_EFFECT_SET = frozenset((EffectInstance(MemoryEffectKind.FREE),))


class MemoryReadEffect(MemoryEffect):
    """
    A trait that signals that an operation always has read side effects.
    """

    @classmethod
    def get_effects(cls, op: Operation) -> AbstractSet[EffectInstance]:
        return _MEMORY_READ_EFFECT_SET


class MemoryWriteEffect(MemoryEffect):
    """
    A trait that signals that an operation always has write side effects.
    """

    @classmethod
    def get_effects(cls, op: Operation) -> AbstractSet[EffectInstance]:
        return _MEMORY_WRITE_EFFECT_SET


class MemoryAllocEffect(MemoryEffect):
    """
    A trait that signals that an operation always has alloc side effects.
    """

    @classmethod
    def get_effects(cls, op: Operation) -> AbstractSet[EffectInstance]:
        return _MEMORY_ALLOC_EFFECT_SET


class MemoryFreeEffect(MemoryEffect):
    """
    A trait that signals that an operation always has deallocation side effects.
    """

    @classmethod
    def get_effects(cls, op: Operation) -> AbstractSet[EffectInstance]:
        return _MEMORY_FREE_EFFECT_SET


class RecursiveMemoryEffect(MemoryEffect):
    """
    A trait that signals that an operation has the side effects of its contained
    operations.
    """

    @classmethod
    def get_effects(cls, op: Operation) -> AbstractSet[EffectInstance] | None:
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
    In MLIR, Pure is NoMemoryEffect + AlwaysSpeculatable.
    """


class Commutative(OpTrait):
    """
    A trait that signals that an operation is commutative.
    """


class HasInsnRepresentation(OpTrait, abc.ABC):
    """
    A trait providing information on how to encode an operation using a .insn assembler directive.

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


SuccessorOperand: TypeAlias = tuple["Operation", int]
"""
A successor operand, identified by its owner operation and operand index. Stands in for MLIR's
`OpOperand *`: xDSL's `Use` hashes by identity, so it cannot be rebuilt to look up a mapping.
"""


@dataclass(frozen=True, init=False)
class OperandRange(Sequence["SSAValue"]):
    """
    The operands of `owner` in `[start, stop)`.

    Like MLIR's `OperandRange`, it keeps track of which operands the values come from, and not just
    the values. The successor mappings need this, as the same value can be passed to several
    operands, e.g. `scf.for %i = %c0 to %n step %c1 iter_args(%a = %c0)`.
    """

    owner: Operation
    start: int
    stop: int

    def __init__(
        self, owner: Operation, start: int = 0, stop: int | None = None
    ) -> None:
        num_operands = len(owner.operands)
        if stop is None:
            stop = num_operands
        if not 0 <= start <= stop <= num_operands:
            raise ValueError(
                f"Operand range [{start}, {stop}) is out of bounds for '{owner.name}', "
                f"which has {num_operands} operands."
            )
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)

    @overload
    def __getitem__(self, index: int) -> SSAValue: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[SSAValue]: ...

    def __getitem__(self, index: int | slice) -> SSAValue | Sequence[SSAValue]:
        return self.owner.operands[self.start : self.stop][index]

    def __iter__(self) -> Iterator[SSAValue]:
        return iter(self.owner.operands[self.start : self.stop])

    def __len__(self) -> int:
        return self.stop - self.start

    @property
    def op_operands(self) -> tuple[SuccessorOperand, ...]:
        """The operands in the range, as `(owner, index)` pairs."""
        return tuple((self.owner, index) for index in range(self.start, self.stop))


RegionBranchSuccessorMapping: TypeAlias = Mapping[
    SuccessorOperand, tuple["SSAValue", ...]
]
"""
A mapping from successor operands to successor inputs.

* A successor operand is an operand of a region branch op or region branch terminator, that is
  forwarded to a successor input.
* A successor input is a block argument of a region or a result of the region branch op, that is
  populated by a successor operand.

The mapping is 1:N. Each successor operand may be forwarded to multiple successor inputs
(because the control flow can dispatch to multiple possible successors). Operands that are not
forwarded at all are not present in the mapping.
"""

RegionBranchInverseSuccessorMapping: TypeAlias = Mapping[
    "SSAValue", tuple[SuccessorOperand, ...]
]
"""
A mapping from successor inputs to the successor operands that populate them. The inverse of
`RegionBranchSuccessorMapping`.
"""


@dataclass(frozen=True)
class RegionSuccessor:
    """
    This class represents a successor of a region. A region successor can either target another
    region or target an ancestor operation (at the moment, limited to the immediate parent
    operation). In the latter case, the control flow branches after/out of the target operation.
    """

    successor: Region | Operation

    @property
    def region(self) -> Region | None:
        """
        The successor region. None if the successor is an operation.

        Implements MLIR `RegionSuccessor::getSuccessor`.
        """
        from xdsl.ir import Region

        return self.successor if isinstance(self.successor, Region) else None

    @property
    def operation(self) -> Operation | None:
        """
        The successor operation. None if the successor is a region.

        Implements MLIR `RegionSuccessor::getSuccessorOp`.
        """
        from xdsl.ir import Operation

        return self.successor if isinstance(self.successor, Operation) else None

    def is_region(self) -> bool:
        """
        Return true if the successor is a region.

        Implements MLIR `RegionSuccessor::isRegion`.
        """
        from xdsl.ir import Region

        return isinstance(self.successor, Region)

    def is_operation(self) -> bool:
        """
        Return true if the successor is an operation.

        Implements MLIR `RegionSuccessor::isOperation`.
        """
        from xdsl.ir import Operation

        return isinstance(self.successor, Operation)


@dataclass(frozen=True, init=False)
class RegionBranchPoint:
    """
    This class represents a point being branched from in the methods of the
    `RegionBranchOpInterface`.

    One can branch from one of two kinds of places:
    * The parent operation (aka the `RegionBranchOpInterface` implementation)
    * A `RegionBranchTerminatorOpInterface` inside a region within the parent operation.

    Construct with the terminator being branched from. Use `RegionBranchPoint.PARENT` to branch
    from the parent operation. Raises `ValueError` if the terminator does not implement
    `RegionBranchTerminatorOpInterface`. The constructor implements MLIR
    `RegionBranchPoint(RegionBranchTerminatorOpInterface)`.
    """

    predecessor: Operation | None

    _interface: RegionBranchTerminatorOpTrait | None = field(compare=False, repr=False)
    """The `RegionBranchTerminatorOpTrait` of `predecessor`."""

    PARENT: ClassVar[RegionBranchPoint]
    """
    The `RegionBranchPoint` representing the parent operation.

    Implements MLIR `RegionBranchPoint::parent`.
    """

    def __init__(self, predecessor: Operation) -> None:
        interface = predecessor.get_trait(RegionBranchTerminatorOpTrait)
        if interface is None:
            raise ValueError(
                f"'{predecessor.name}' does not implement RegionBranchTerminatorOpInterface."
            )
        object.__setattr__(self, "predecessor", predecessor)
        object.__setattr__(self, "_interface", interface)

    def is_parent(self) -> bool:
        """
        Returns true if branching from the parent op.

        Implements MLIR `RegionBranchPoint::isParent`.
        """
        return self.predecessor is None

    @property
    def terminator(self) -> Operation | None:
        """
        The terminator if branching from a region. None otherwise.

        Implements MLIR `RegionBranchPoint::getTerminatorPredecessorOrNull`.
        """
        return self.predecessor

    @property
    def terminator_predecessor_and_trait(
        self,
    ) -> tuple[Operation, RegionBranchTerminatorOpTrait] | None:
        """
        The terminator and its `RegionBranchTerminatorOpTrait` if branching from a
        region. None otherwise.

        Implements MLIR `RegionBranchPoint::getTerminatorPredecessorOrNull`, together with
        `terminator`. MLIR returns a single interface handle that is both the op and its interface
        methods. In xDSL the op and its trait are separate objects, and the trait methods take the
        op as an argument, so this property returns both.
        """
        if self.predecessor is None or self._interface is None:
            return None
        return self.predecessor, self._interface


# Mirrors MLIR's private default constructor: bypasses `__init__`, which rejects `None`.
RegionBranchPoint.PARENT = object.__new__(RegionBranchPoint)
object.__setattr__(RegionBranchPoint.PARENT, "predecessor", None)
object.__setattr__(RegionBranchPoint.PARENT, "_interface", None)


@dataclass(frozen=True)
class InvocationBounds:
    """
    This class represents upper and lower bounds on the number of times a region of a
    `RegionBranchOpInterface` can be invoked. The lower bound is at least zero, but the upper bound
    may not be known.
    """

    lower: int
    """
    The minimum number of times the successor region will be invoked.

    Implements MLIR `InvocationBounds::getLowerBound`.
    """

    upper: int | None
    """
    The maximum number of times the successor region will be invoked or `None` if an upper bound
    is not known.

    Implements MLIR `InvocationBounds::getUpperBound`.
    """

    UNKNOWN: ClassVar[InvocationBounds]
    """
    The unknown invocation bounds, i.e., there is no information on how many times a region may
    be invoked.

    Implements MLIR `InvocationBounds::getUnknown`.
    """

    def __post_init__(self) -> None:
        """Implements the checks of MLIR `InvocationBounds` constructor."""
        if self.lower < 0:
            raise ValueError(
                f"Invocation lower bound must be at least 0, got {self.lower}."
            )
        if self.upper is not None and self.upper < self.lower:
            raise ValueError(
                f"Invocation upper bound ({self.upper}) cannot be less than lower bound "
                f"({self.lower})."
            )


InvocationBounds.UNKNOWN = InvocationBounds(0, None)


_StopConditionFn: TypeAlias = Callable[["Region", Sequence[bool]], bool]
"""
Stop condition for `RegionBranchOpTrait._traverse_region_graph`. The traversal is interrupted if
this function returns True for a successor region. The first parameter is the successor region. The
second parameter indicates all already visited regions.
"""


class RegionBranchOpTrait(OpTrait):
    """
    This interface provides information for region-holding operations that exhibit branching
    behavior between held regions. It models the control flow edges between regions (and between
    the op and its regions), as well as the data flow (value propagation) that occurs along those
    control flow edges.

    This interface is meant to model well-defined cases of control-flow and value propagation,
    where what occurs along control-flow edges is assumed to be side-effect free.

    A "region branch point" indicates the point from which a branch (edge) originates. It can
    indicate:
    1. A `RegionBranchTerminatorOpInterface` terminator in any of the immediately nested regions
       of this op.
    2. `RegionBranchPoint.PARENT`: the branch originates from outside of the op, i.e., when
       first executing this op.

    When branching from a region branch point to a region successor, the "successor operands" to
    be forwarded from the region branch point can be specified with `entry_successor_operands`
    / `RegionBranchTerminatorOpInterface.successor_operands`.

    A "region successor" indicates the target of a branch. It can indicate:
    1. A region of this op.
    2. A parent operation, i.e., the control flow leaves/resumes after that op.

    The SSA values to which successor operands are forwarded are called "successor inputs".

    By default, successor operands and successor block arguments/successor results must have the
    same type. `are_types_compatible` can be implemented to allow non-equal types.

    Note: This interface works in conjunction with `RegionBranchTerminatorOpInterface`. All
    immediately nested block terminators that model branching between regions must implement the
    `RegionBranchTerminatorOpInterface`. Otherwise, analyses/transformations may miss control flow
    edges and produce incorrect results. Not every block terminator is necessarily a region branch
    terminator: e.g., in the presence of unstructured control flow, a block terminator could
    indicate a branch to a different block within the same region.

    Example:

    ```
    %r = scf.for %iv = %lb to %ub step %step iter_args(%a = %b)
        -> tensor<5xf32> {
      ...
      scf.yield %c : tensor<5xf32>
    }
    ```

    `scf.for` has one region. There are two region branch points with two identical region
    successors:
    * parent => op(%r), region0(%a)
    * `scf.yield` => op(%r), region0(%a)

    `%a` and `%r` are successor inputs. `%b` is an entry successor operand. `%c` is a successor
    operand.
    """

    @classmethod
    def entry_successor_operands(
        cls,
        op: Operation,
        successor: RegionSuccessor,
    ) -> OperandRange:
        """
        Returns the operands of this operation that are forwarded to the successor inputs when
        branching to `successor`. `successor` is guaranteed to be among the successors that are
        returned by `entry_successor_regions` /
        `successor_regions(op, RegionBranchPoint.PARENT)`.

        Example: In the above example, this method returns the operand `%b` of the `scf.for` op,
        regardless of the value of `successor`. I.e., this op always forwards the same operands,
        regardless of whether the loop has 0 or more iterations.
        """
        return OperandRange(op, 0, 0)

    @classmethod
    def entry_successor_regions(
        cls,
        op: Operation,
        operands: Sequence[Attribute | None],
    ) -> Sequence[RegionSuccessor]:
        """
        Returns all potential region successors when first executing the op.

        Unlike `successor_regions`, this method also receives the constant operands of this op
        (one entry per operand, `None` if the operand has no/unknown constant value). The
        implementation may use this information to filter out successors. By default, it simply
        dispatches to `successor_regions`.

        Note: The control flow does not necessarily have to enter any region of this op.

        Example: In the above example, this method may return two region successors: the single
        region of the `scf.for` op and the `scf.for` operation (that implements this interface). If
        `%lb`, `%ub`, `%step` are constants and it can be determined the loop does not have any
        iterations, this method may choose to return only this operation. Similarly, if it can be
        determined that the loop has at least one iteration, this method may choose to return only
        the region of the loop.
        """
        return cls.successor_regions(op, RegionBranchPoint.PARENT)

    @classmethod
    @abc.abstractmethod
    def successor_regions(
        cls, op: Operation, point: RegionBranchPoint
    ) -> Sequence[RegionSuccessor]:
        """
        Returns all potential region successors when branching from `point`. These are the
        regions that may be selected during the flow of control.

        When `point = RegionBranchPoint.PARENT`, this method returns the region successors when
        entering the operation. Otherwise, this method returns the successor regions when branching
        from the region indicated by `point`.

        Example: In the above example, this method returns the region of the `scf.for` and the
        `scf.for` op itself for either region branch point. An implementation may choose to filter
        out region successors when it is statically known (e.g., by examining the operands of this
        op) that those successors are not branched to.
        """
        raise NotImplementedError()

    @classmethod
    def successor_regions_from_region(
        cls, op: Operation, region: Region
    ) -> Sequence[RegionSuccessor]:
        """
        Returns all potential region successors when branching from any terminator in
        `region`.
        """
        regions: list[RegionSuccessor] = []
        for block in region.blocks:
            terminator = block.last_op
            if terminator is not None and terminator.has_trait(
                RegionBranchTerminatorOpTrait
            ):
                regions.extend(cls.successor_regions(op, RegionBranchPoint(terminator)))
        return regions

    @classmethod
    def successor_inputs(
        cls,
        op: Operation,
        successor: RegionSuccessor,
    ) -> Sequence[SSAValue]:
        """
        Return all successor inputs for the given region successor. If the given region
        successor is a region, then the returned values are block arguments. Otherwise, if the
        given region successor is an operation, the returned values are op results.
        """
        return ()

    @classmethod
    def predecessors(
        cls, op: Operation, successor: RegionSuccessor
    ) -> Sequence[RegionBranchPoint]:
        """Returns all potential branching points (predecessors) for a given region successor."""
        return [
            point
            for point in cls.all_region_branch_points(op)
            if successor in cls.successor_regions(op, point)
        ]

    @classmethod
    def predecessor_values(
        cls, op: Operation, successor: RegionSuccessor, index: int
    ) -> Sequence[SSAValue]:
        """
        Returns all potential values across all (predecessors) for a given successor input,
        modeled by its index (its position in the list of values).
        """
        predecessors = cls.predecessors(op, successor)
        predecessor_values: list[SSAValue] = []
        for predecessor in predecessors:
            if predecessor.is_parent():
                predecessor_values.append(
                    cls.entry_successor_operands(op, successor)[index]
                )
                continue
            predecessor_values.append(
                cls.successor_operands(op, predecessor, successor)[index]
            )

        return predecessor_values

    @classmethod
    def region_invocation_bounds(
        cls,
        op: Operation,
        operands: Sequence[Attribute | None],
    ) -> Sequence[InvocationBounds]:
        """
        Returns the minimum and maximum number of times this operation will invoke the attached
        regions (assuming the regions yield normally, i.e. do not abort or invoke an infinite
        loop). The minimum number of invocations is at least 0. If the maximum number of
        invocations cannot be statically determined, then it will be set to
        `InvocationBounds.UNKNOWN`.

        This method also passes along the constant operands of this op. `operands` contains an
        entry for every operand of this op, with `None` if the operand has no constant value.

        This method may be called speculatively on operations where the provided operands are not
        necessarily the same as the operation's current operands. This may occur in analyses that
        wish to determine "what would be the region invocations if these were the operands?"
        """
        return [InvocationBounds.UNKNOWN] * len(op.regions)

    @classmethod
    def are_types_compatible(
        cls, op: Operation, lhs: Attribute, rhs: Attribute
    ) -> bool:
        """
        This method is called to compare types along control-flow edges. By default, the types
        are checked as equal.
        """
        return lhs == rhs

    @override
    def verify(self, op: Operation) -> None:
        from xdsl.irdl import IRDLOperation

        # MLIR rejects ops declared with zero regions, so an op with variadic regions
        # may still have none at runtime.
        if isinstance(op, IRDLOperation) and not op.get_irdl_definition().regions:
            raise VerifyException(
                f"'{op.name}' implements RegionBranchOpInterface but defines no regions."
            )

        self._verify_region_branch_op_interface(op)

    @classmethod
    def is_repetitive_region(cls, op: Operation, index: int) -> bool:
        """
        Return True if control flow originating from the region at `index` may eventually branch
        back to the same region (maybe after passing through other regions).
        """
        region = op.regions[index]
        return cls._is_region_reachable(region, region)

    @classmethod
    def has_loop(cls, op: Operation) -> bool:
        """
        Return True if there is a loop in the region branching graph. Only reachable regions
        (starting from the entry regions) are considered.
        """
        region_indices = {region: index for index, region in enumerate(op.regions)}
        for successor in cls.successor_regions(op, RegionBranchPoint.PARENT):
            if (region := successor.region) is not None:
                has_loop = cls._traverse_region_graph(
                    region,
                    # Interrupt traversal if the region was already visited.
                    lambda next_region, visited: visited[region_indices[next_region]],
                )
                if has_loop:
                    return True

        return False

    @classmethod
    def successor_operands(
        cls, op: Operation, src: RegionBranchPoint, dest: RegionSuccessor
    ) -> OperandRange:
        """
        Return the successor operands from the source branch point to the destination region
        successor.

        If the branch point is the parent op, this function returns the entry successor operands of
        this op. Otherwise, it returns the successor operands of the respective terminator.
        """
        if (terminator := src.terminator_predecessor_and_trait) is None:
            return cls.entry_successor_operands(op, dest)
        terminator_op, trait = terminator
        return trait.successor_operands(terminator_op, dest)

    @classmethod
    def non_successor_inputs(
        cls, op: Operation, successor: RegionSuccessor
    ) -> Sequence[SSAValue]:
        """
        Return all non-successor inputs for the given region successor. If the successor is a
        region, these are the block arguments not forwarded to; if it is the parent, these are the
        results not forwarded to.
        """
        from xdsl.ir import BlockArgument, OpResult

        results: list[SSAValue]
        if (successor_op := successor.operation) is not None:
            # successor is an Operation
            results = list(successor_op.results)
        else:
            # successor is a Region
            region = successor.region
            assert region is not None
            entry = region.first_block
            results = [] if entry is None else list(entry.args)

        successor_inputs = cls.successor_inputs(op, successor)
        if successor_inputs:
            first = successor_inputs[0]
            if successor.is_operation():
                assert isinstance(first, OpResult)
            else:
                assert isinstance(first, BlockArgument)
            input_begin = first.index
            del results[input_begin : input_begin + len(successor_inputs)]
        return results

    @classmethod
    def successor_operand_input_mapping(
        cls, op: Operation, src: RegionBranchPoint | None = None
    ) -> RegionBranchSuccessorMapping:
        """
        Build a mapping from successor operands to successor inputs. Each successor operand could
        be forwarded to multiple successor inputs. Operands that are not forwarded are not added to
        the mapping. Unless a specific region branch point is specified, this function takes into
        account all possible region branch points.
        """
        mapping: dict[SuccessorOperand, list[SSAValue]] = {}
        if src is not None:
            cls._successor_operand_input_mapping(op, mapping, src)
        else:
            # No region branch point specified: populate the mapping for all possible region
            # branch points.
            for branch_point in cls.all_region_branch_points(op):
                cls._successor_operand_input_mapping(op, mapping, branch_point)
        return {operand: tuple(inputs) for operand, inputs in mapping.items()}

    @classmethod
    def successor_input_operand_mapping(
        cls, op: Operation
    ) -> RegionBranchInverseSuccessorMapping:
        """
        Build a mapping from successor inputs to successor operands. This is the same as
        `successor_operand_input_mapping`, but inverted.
        """
        inverse: dict[SSAValue, list[SuccessorOperand]] = {}
        for operand, inputs in cls.successor_operand_input_mapping(op).items():
            for input_ in inputs:
                inverse.setdefault(input_, []).append(operand)
        return {input_: tuple(operands) for input_, operands in inverse.items()}

    @classmethod
    def all_region_branch_points(cls, op: Operation) -> Sequence[RegionBranchPoint]:
        """
        Return all possible region branch points: the region branch op itself and all region
        branch terminators.
        """
        points = [RegionBranchPoint.PARENT]
        for region in op.regions:
            for block in region.blocks:
                terminator = block.last_op
                if terminator is not None and terminator.has_trait(
                    RegionBranchTerminatorOpTrait
                ):
                    points.append(RegionBranchPoint(terminator))
        return points

    @staticmethod
    def _edge_description(
        op: Operation, point: RegionBranchPoint, successor: RegionSuccessor
    ) -> str:
        """
        Return a description of the control flow edge from `point` to `successor` of `op`, for
        verifier error messages.
        """
        terminator = point.terminator
        source = "parent" if terminator is None else f"Operation '{terminator.name}'"
        if (region := successor.region) is not None:
            target = f"Region #{op.get_region_index(region)}"
        else:
            successor_op = successor.operation
            assert successor_op is not None
            target = f"Operation '{successor_op.name}'"
        return f"'{op.name}': along control flow edge from {source} to {target}"

    @classmethod
    def _verify_region_branch_op_interface(cls, op: Operation) -> None:
        """Verify that types match along control flow edges described by the given op."""
        # Verify all control flow edges from region branch points to region successors.
        for point in cls.all_region_branch_points(op):
            for successor in cls.successor_regions(op, point):
                # Verify number of successor operands and successor inputs.
                succ_operands = cls.successor_operands(op, point, successor)
                succ_inputs = cls.successor_inputs(op, successor)
                if len(succ_operands) != len(succ_inputs):
                    edge = cls._edge_description(op, point, successor)
                    raise VerifyException(
                        f"{edge}: region branch point has {len(succ_operands)} operands, but "
                        f"region successor needs {len(succ_inputs)} inputs"
                    )

                # Verify that the types are compatible.
                for index, (succ_operand, succ_input) in enumerate(
                    zip(succ_operands, succ_inputs, strict=True)
                ):
                    if not cls.are_types_compatible(
                        op, succ_operand.type, succ_input.type
                    ):
                        edge = cls._edge_description(op, point, successor)
                        raise VerifyException(
                            f"{edge}: successor operand type #{index} {succ_operand.type} should "
                            f"match successor input type #{index} {succ_input.type}"
                        )

    @staticmethod
    def _region_successors(region: Region) -> list[Region]:
        """
        Return the region successors of every region branch terminator in `region`. Exits to the
        parent op are skipped, since they cannot lead back into a region.
        """
        successors: list[Region] = []
        for block in region.blocks:
            terminator = block.last_op
            if terminator is None:
                continue
            trait = terminator.get_trait(RegionBranchTerminatorOpTrait)
            if trait is None:
                continue
            operand_attributes: list[Attribute | None] = [None] * len(
                terminator.operands
            )
            for successor in trait.successor_regions(terminator, operand_attributes):
                if (successor_region := successor.region) is not None:
                    successors.append(successor_region)
        return successors

    @classmethod
    def _traverse_region_graph(
        cls, begin: Region, stop_condition_fn: _StopConditionFn
    ) -> bool:
        """
        Traverse the region graph starting at `begin`. The traversal is interrupted if
        `stop_condition_fn` evaluates to True for a successor region. In that case, this function
        returns True. Otherwise, if the traversal was not interrupted, this function returns
        False.
        """
        op = begin.parent_op()
        assert op is not None
        assert op.has_trait(RegionBranchOpTrait)

        region_indices = {region: index for index, region in enumerate(op.regions)}
        visited = [False] * len(op.regions)
        visited[region_indices[begin]] = True

        # Retrieve all successors of the region and enqueue them in the worklist.
        worklist = cls._region_successors(begin)

        # Process all regions in the worklist via DFS.
        while worklist:
            next_region = worklist.pop()
            if stop_condition_fn(next_region, visited):
                return True
            next_index = region_indices[next_region]
            if visited[next_index]:
                continue
            visited[next_index] = True
            worklist.extend(cls._region_successors(next_region))

        return False

    @classmethod
    def _is_region_reachable(cls, begin: Region, r: Region) -> bool:
        """
        Return True if region `r` is reachable from region `begin` according to the
        `RegionBranchOpInterface` (by taking a branch).
        """
        assert begin.parent_op() is r.parent_op(), (
            "expected that both regions belong to the same op"
        )
        return cls._traverse_region_graph(
            begin, lambda next_region, _visited: next_region is r
        )

    @classmethod
    def _successor_operand_input_mapping(
        cls,
        op: Operation,
        mapping: dict[SuccessorOperand, list[SSAValue]],
        src: RegionBranchPoint,
    ) -> None:
        """
        Populate `mapping` with the successor operands forwarded from `src`, each mapped to the
        successor inputs it is forwarded to.
        """
        successors = cls.successor_regions(op, src)
        for dst in successors:
            operands = cls.successor_operands(op, src, dst)
            inputs = cls.successor_inputs(op, dst)
            assert len(operands) == len(inputs), (
                "expected the same number of operands and inputs"
            )
            for operand, input_ in zip(operands.op_operands, inputs, strict=True):
                mapping.setdefault(operand, []).append(input_)


class RegionBranchTerminatorOpTrait(OpTrait):
    """
    This interface provides information for branching terminator operations in the presence of a
    parent `RegionBranchOpInterface` implementation. It acts as a marker for valid region branch
    points and specifies which operands are passed to which region successor.

    Note: If an operation does not implement the `RegionBranchTerminatorOpInterface`, then that op
    has no region successors. (However, there may be other block terminators in the same region
    that implement the `RegionBranchTerminatorOpInterface`, so the enclosing region may have region
    successors.)
    """

    @classmethod
    def successor_operands(
        cls,
        op: Operation,
        successor: RegionSuccessor,
    ) -> OperandRange:
        """Returns the operands passed to `successor`. By default, all operands of `op`."""
        return OperandRange(op)

    @classmethod
    def successor_regions(
        cls,
        op: Operation,
        operands: Sequence[Attribute | None],
    ) -> Sequence[RegionSuccessor]:
        """
        Returns all potential region successors that are branched to after this terminator based
        on the given constant operands.

        This method also receives the constant operands of this op (one entry per operand, `None`
        if the operand has no/unknown constant value). The implementation may use this
        information to filter out successors. By default, it simply dispatches to the parent
        `RegionBranchOpInterface`'s `successor_regions` implementation.
        """
        parent = op.parent_op()
        if parent is None:
            raise ValueError(f"'{op.name}' has no parent op.")
        trait = parent.get_trait(RegionBranchOpTrait)
        if trait is None:
            raise ValueError(
                f"'{parent.name}', parent of '{op.name}', does not implement "
                "RegionBranchOpInterface."
            )
        return trait.successor_regions(parent, RegionBranchPoint(op))

    @override
    def verify(self, op: Operation) -> None:
        if not op.has_trait(IsTerminator):
            raise VerifyException(f"Expected operation '{op.name}' to be a terminator.")
        if op.results:
            raise VerifyException(
                f"Expected operation '{op.name}' to have zero results."
            )
        if op.successors:
            raise VerifyException(
                f"Expected operation '{op.name}' to have zero successors."
            )
