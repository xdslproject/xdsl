from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from typing_extensions import Self

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
from xdsl.traits import EffectInstance, MemoryEffect, MemoryEffectKind, Resource


@dataclass(frozen=True)
class RegisterType(ParametrizedAttribute, TypeAttribute, ABC):
    """
    An abstract register type for target ISA-specific dialects.

    Registers have a name, as used in assembly, and an index as used in the binary
    encoding.

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

    @classmethod
    def unallocated(cls) -> Self:
        """
        Returns an unallocated register of this type.
        """
        return cls(NoneAttr())

    @classmethod
    def from_index(cls, index: int) -> Self:
        return cls(IntAttr(index))

    @property
    def is_allocated(self) -> bool:
        """Returns true if the register is allocated, otherwise false"""
        return not isinstance(self.index, NoneAttr)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        if parser.parse_optional_punctuation("<"):
            index = parser.parse_integer()
            parser.parse_punctuation(">")
            return (IntAttr(index),)
        else:
            return (NoneAttr(),)

    def print_parameters(self, printer: Printer) -> None:
        if isinstance(self.index, IntAttr):
            with printer.in_angle_brackets():
                printer.print_int(self.index.data)

    @classmethod
    def allocatable_registers(cls) -> Sequence[Self]:
        """
        Registers of this type that can be used for register allocation.
        """
        return ()

    @classmethod
    def infinite_register(cls, index: int) -> Self:
        """
        Provide the register at the given index in the "infinite" register set.
        Index must be positive.
        """
        assert index >= 0, f"Infinite index must be positive, got {index}."
        index_attr = IntAttr(~index)
        res = cls(index_attr)
        return res


class NamedRegisterType(RegisterType, ABC):
    @classmethod
    def from_index(cls, index: int) -> Self:
        if index >= 0 and index not in cls.abi_name_by_index():
            raise ValueError(f"Invalid index {index} for register class {cls.name}")
        return cls(IntAttr(index))

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

    @property
    def register_name_str(self) -> str:
        if not isinstance(index_attr := self.index, IntAttr):
            return ""

        index = index_attr.data
        if 0 <= index:
            return self.abi_name_by_index()[index]
        else:
            return self.infinite_register_prefix() + str(~index)

    @property
    def register_name(self) -> StringAttr:
        return StringAttr(self.register_name_str)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        if parser.parse_optional_punctuation("<"):
            pos = parser.pos
            name = parser.parse_identifier()
            parser.parse_punctuation(">")
            try:
                params = cls._parameters_from_name(StringAttr(name))
            except ValueError as e:
                parser.raise_error(f"{e}", pos)
        else:
            params = (NoneAttr(),)

        return params

    def print_parameters(self, printer: Printer) -> None:
        register_name = self.register_name_str
        if register_name:
            with printer.in_angle_brackets():
                printer.print_string(register_name)

    @classmethod
    def _parameters_from_name(
        cls, register_name: StringAttr
    ) -> tuple[IntAttr | NoneAttr]:
        """
        Returns the parameter list required to construct a register instance from the given register_name.
        """
        if not register_name.data:
            return (NoneAttr(),)

        index = cls.index_by_name().get(register_name.data)
        if index is None:
            # Try to decode as infinite register
            prefix = cls.infinite_register_prefix()
            if register_name.data.startswith(prefix):
                suffix = register_name.data[len(prefix) :]
                # infinite registers go from -1 to -inf
                try:
                    index = ~int(suffix)
                except ValueError:
                    index = None
            else:
                index = None

        if index is None:
            raise ValueError(
                f"Invalid register name {register_name.data} for register type "
                f"{cls.name}."
            )

        # Raise verification error instead
        index_attr = IntAttr(index)
        return (index_attr,)

    @classmethod
    def from_name(cls, register_name: StringAttr | str) -> Self:
        if not isinstance(register_name, StringAttr):
            register_name = StringAttr(register_name)
        return cls(*cls._parameters_from_name(register_name))

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
