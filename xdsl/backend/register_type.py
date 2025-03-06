from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from typing_extensions import Self

from xdsl.dialects.builtin import (
    IntAttr,
    NoneAttr,
    StringAttr,
)
from xdsl.ir import (
    Attribute,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.irdl import ParameterDef
from xdsl.parser import AttrParser
from xdsl.printer import Printer


class RegisterType(ParametrizedAttribute, TypeAttribute, ABC):
    """
    An abstract register type for target ISA-specific dialects.
    """

    index: ParameterDef[IntAttr | NoneAttr]
    register_name: ParameterDef[StringAttr]

    def __init__(self, index: IntAttr | NoneAttr, register_name: StringAttr):
        super().__init__((index, register_name))

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
        index = cls.abi_index_by_name().get(register_name.data)
        index_attr = NoneAttr() if index is None else IntAttr(index)
        return index_attr, register_name

    @classmethod
    def from_name(cls, register_name: StringAttr | str) -> Self:
        if not isinstance(register_name, StringAttr):
            register_name = StringAttr(register_name)
        return cls(*cls._parameters_from_name(register_name))

    @property
    def is_allocated(self) -> bool:
        """Returns true if the register is allocated, otherwise false"""
        return bool(self.register_name.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        if parser.parse_optional_punctuation("<"):
            name = parser.parse_identifier()
            parser.parse_punctuation(">")
        else:
            name = ""
        return cls._parameters_from_name(StringAttr(name))

    def print_parameters(self, printer: Printer) -> None:
        if self.register_name.data:
            with printer.in_angle_brackets():
                printer.print_string(self.register_name.data)

    def verify(self) -> None:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def instruction_set_name(cls) -> str:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        raise NotImplementedError()

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
        """
        register_name = cls.infinite_register_prefix() + str(index)
        res = cls.from_name(register_name)
        assert isinstance(res.index, NoneAttr), (
            f"Invalid 'infinite' register name: {register_name} clashes with finite register set"
        )
        return res
