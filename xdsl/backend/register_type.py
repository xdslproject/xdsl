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
from xdsl.utils.exceptions import VerifyException


@dataclass(frozen=True)
class RegisterType(ParametrizedAttribute, TypeAttribute, ABC):
    """
    An abstract register type for target ISA-specific dialects.

    Registers have a name, as used in assembly, and an index as used in the binary
    encoding.

    Some approaches for register allocation have stages where values are assigned to a
    fixed set of registers that is distinct from the registers that exist on a target
    platform, to separate the graph coloring from roles of registers in the target ABI.
    In orger to support this scenario, negative indices are allowed in the index,
    denoting an infinite register set without any representation in the ABI.
    These are printed with a prefix as defined by the `infinite_register_prefix`
    class method, which must not be a prefix of any of the register names defined by
    the `index_by_name` class method.
    """

    index: IntAttr | NoneAttr
    register_name: StringAttr

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
        return cls(NoneAttr(), StringAttr(""))

    @classmethod
    def _parameters_from_name(
        cls, register_name: StringAttr
    ) -> tuple[IntAttr | NoneAttr, StringAttr]:
        """
        Returns the parameter list required to construct a register instance from the given register_name.
        """
        if not register_name.data:
            return NoneAttr(), register_name
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

        # Raise verification error instead
        index_attr = NoneAttr() if index is None else IntAttr(index)
        return index_attr, register_name

    @classmethod
    def from_name(cls, register_name: StringAttr | str) -> Self:
        if not isinstance(register_name, StringAttr):
            register_name = StringAttr(register_name)
        return cls(*cls._parameters_from_name(register_name))

    @classmethod
    def from_index(cls, index: int) -> Self:
        if index < 0:
            return cls.infinite_register(~index)
        name = cls.abi_name_by_index()[index]
        return cls(IntAttr(index), StringAttr(name))

    @property
    def is_allocated(self) -> bool:
        """Returns true if the register is allocated, otherwise false"""
        return bool(self.register_name.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        if parser.parse_optional_punctuation("<"):
            name = parser.parse_identifier()
            parser.parse_punctuation(">")
            params = cls._parameters_from_name(StringAttr(name))
        else:
            params = (NoneAttr(), StringAttr(""))

        return params

    def print_parameters(self, printer: Printer) -> None:
        if self.register_name.data:
            with printer.in_angle_brackets():
                printer.print_string(self.register_name.data)

    def verify(self) -> None:
        name = self.register_name.data
        expected_index = type(self).index_by_name().get(name)

        if isinstance(self.index, NoneAttr):
            if not name:
                # Unallocated, expect NoneAttr
                return

            if expected_index is None:
                raise VerifyException(
                    f"Invalid register name {name} for register type {self.name}."
                )
            else:
                raise VerifyException(
                    f"Missing index for register {name}, expected {expected_index}."
                )

        if not name:
            raise VerifyException(
                f"Invalid index {self.index.data} for unallocated register."
            )

        if expected_index is not None:
            # Normal registers
            if expected_index == self.index.data:
                return

            raise VerifyException(
                f"Invalid index {self.index.data} for register {name}, expected {expected_index}."
            )

        infinite_register_name = self.infinite_register_prefix() + str(~self.index.data)
        if name == infinite_register_name:
            return

        raise VerifyException(f"Invalid index {self.index.data} for register {name}.")

    @classmethod
    def allocatable_registers(cls) -> Sequence[Self]:
        """
        Registers of this type that can be used for register allocation.
        """
        return ()

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
        Index must be positive.
        """
        assert index >= 0, f"Infinite index must be positive, got {index}."
        register_name = cls.infinite_register_prefix() + str(index)
        assert register_name not in cls.index_by_name(), (
            f"Invalid 'infinite' register name: {register_name} clashes with finite register set"
        )
        index_attr = IntAttr(~index)
        res = cls(index_attr, StringAttr(register_name))
        return res


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
