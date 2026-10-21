from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Self

from xdsl.dialects.builtin import (
    IntAttr,
    NoneAttr,
    StringAttr,
)
from xdsl.ir import (
    Attribute,
    Operation,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import (
    EffectInstance,
    MemoryEffect,
    MemoryEffectKind,
    Resource,
    get_effects,
)
from xdsl.utils.exceptions import VerifyException


@dataclass(frozen=True)
class RegisterType(ParametrizedAttribute, TypeAttribute, ABC):
    """
    An abstract register type for target ISA-specific dialects.

    Registers store an index as used in the binary encoding. Their assembly names
    are derived from the index, so aliases identify the same register.

    Some approaches for register allocation have stages where values are assigned to a
    fixed set of registers that is distinct from the registers that exist on a target
    platform, to separate the graph coloring from roles of registers in the target ABI.
    In order to support this scenario, negative indices are allowed in the index,
    denoting an infinite register set without any representation in the ABI.
    These are printed with a prefix as defined by the `infinite_register_prefix`
    class method, which must not be a prefix of any of the register names defined by
    the `index_by_name` class method.
    """

    index: IntAttr | NoneAttr

    @property
    def register_name(self) -> StringAttr:
        """The canonical assembly name, or an empty string if unallocated."""
        if isinstance(self.index, NoneAttr):
            return StringAttr("")
        if self.index.data < 0:
            return StringAttr(self.infinite_register_prefix() + str(~self.index.data))
        return StringAttr(self.abi_name_by_index()[self.index.data])

    def __init_subclass__(cls) -> None:
        # Detect register names clashing with the infinite register prefix
        try:
            prefix = cls.infinite_register_prefix()
            names = cls.index_by_name()
        except NotImplementedError:
            # Skip for abstract subclasses
            return

        clashing_register_names = tuple(
            register_name for register_name in names if register_name.startswith(prefix)
        )

        if clashing_register_names:
            raise ValueError(
                f"Infinite register prefix '{prefix}' clashes with register names "
                f"{list(clashing_register_names)}."
            )

    @classmethod
    def unallocated(cls) -> Self:
        """
        Returns an unallocated register of this type.
        """
        return cls(NoneAttr())

    @classmethod
    def _index_from_name(cls, name: str) -> IntAttr | NoneAttr:
        """Resolve an assembly name to its register index."""
        if not name:
            return NoneAttr()
        if (index := cls.index_by_name().get(name)) is not None:
            return IntAttr(index)

        prefix = cls.infinite_register_prefix()
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            try:
                infinite_index = int(suffix)
            except ValueError:
                pass
            else:
                if infinite_index >= 0 and suffix == str(infinite_index):
                    return IntAttr(~infinite_index)

        raise ValueError(f"Invalid register name {name} for register type {cls.name}.")

    @classmethod
    def from_name(cls, register_name: StringAttr | str) -> Self:
        if isinstance(register_name, StringAttr):
            register_name = register_name.data
        return cls(cls._index_from_name(register_name))

    @classmethod
    def from_index(cls, index: int) -> Self:
        if index >= 0 and index not in cls.abi_name_by_index():
            raise ValueError(f"Invalid index {index} for register type {cls.name}.")
        return cls(IntAttr(index))

    @property
    def is_allocated(self) -> bool:
        """Returns true if the register is allocated, otherwise false"""
        return isinstance(self.index, IntAttr)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        if parser.parse_optional_punctuation("<"):
            start_pos = parser.pos
            name = parser.parse_identifier()
            end_pos = parser.pos
            parser.parse_punctuation(">")
            try:
                return (cls._index_from_name(name),)
            except ValueError as e:
                parser.raise_error(str(e), start_pos, end_pos)
        return (NoneAttr(),)

    def print_parameters(self, printer: Printer) -> None:
        if self.is_allocated:
            with printer.in_angle_brackets():
                printer.print_string(self.register_name.data)

    def verify(self) -> None:
        if (
            isinstance(self.index, IntAttr)
            and self.index.data >= 0
            and self.index.data not in self.abi_name_by_index()
        ):
            raise VerifyException(
                f"Invalid index {self.index.data} for register type {self.name}."
            )

    @classmethod
    def allocatable_registers(cls) -> Sequence[Self]:
        """
        Registers of this type that can be used for register allocation.
        """
        return ()

    @classmethod
    def register_pool_key(cls) -> str:
        """
        Pool key used when allocating unallocated values of this type and for infinite
        spill registers. Defaults to the dialect type name (``cls.name``).
        """
        return cls.name

    @classmethod
    @abstractmethod
    def index_by_name(cls) -> dict[str, int]:
        raise NotImplementedError()

    # This class variable is created and exclusively accessed in `abi_name_by_index`.
    # _ABI_NAME_BY_INDEX: ClassVar[dict[int, str]]

    @classmethod
    def abi_name_by_index(cls) -> dict[int, str]:
        """
        Returns a mapping from ABI register indices to their names.
        """
        if hasattr(cls, "_ABI_NAME_BY_INDEX"):
            return cls._ABI_NAME_BY_INDEX

        result = {i: n for n, i in cls.index_by_name().items()}
        cls._ABI_NAME_BY_INDEX = result
        return result

    @classmethod
    @abstractmethod
    def infinite_register_prefix(cls) -> str:
        """
        Provide the prefix for the name for a register at the given index in the
        "infinite" register set.
        For a prefix `x`, the name of the first infinite register will be `x0`.
        """
        raise NotImplementedError()

    @classmethod
    def infinite_register(cls, index: int) -> Self:
        """
        Provide the register at the given index in the "infinite" register set.
        Index must be nonnegative.
        """
        if index < 0:
            raise ValueError(f"Infinite index must be nonnegative, got {index}.")
        return cls(IntAttr(~index))


@dataclass(frozen=True)
class RegisterResource(Resource):
    register: RegisterType

    def name(self) -> str:
        return f"<Register {self.register}>"


class RegisterAllocatedMemoryEffect(MemoryEffect):
    """
    An assembly operation that only has side-effect if some registers are allocated to
    it.
    """

    @classmethod
    def get_effects(cls, op: Operation) -> set[EffectInstance]:
        effects = set[EffectInstance]()
        for result in op.results:
            if isinstance(r := result.type, RegisterType) and r.is_allocated:
                effects.add(
                    EffectInstance(MemoryEffectKind.WRITE, resource=RegisterResource(r))
                )
        for operand in op.operands:
            if isinstance(r := operand.type, RegisterType) and r.is_allocated:
                effects.add(
                    EffectInstance(MemoryEffectKind.READ, resource=RegisterResource(r))
                )
        return effects

    @staticmethod
    def iter_used_registers(op: Operation) -> Iterator[RegisterType]:
        """
        Iterator over the registers read from or written to according to the operation's
        memory effects with resource `RegisterResource`.
        """
        effects = get_effects(op)
        if effects is not None:
            yield from (
                resource.register
                for effect in effects
                if isinstance(resource := effect.resource, RegisterResource)
            )

        # Include registers from deprecated call:
        import warnings

        from xdsl.backend.register_allocatable import RegisterAllocatableOperation

        if isinstance(op, RegisterAllocatableOperation) and (
            type(op).iter_used_registers  # pyright: ignore[reportDeprecated]
            is not RegisterAllocatableOperation.iter_used_registers  # pyright: ignore[reportDeprecated]
        ):
            warnings.warn(
                "RegisterAllocatableOperation.iter_used_registers is "
                "deprecated, use register effects instead.",
                DeprecationWarning,
            )
            yield from op.iter_used_registers()  # pyright: ignore[reportDeprecated]
