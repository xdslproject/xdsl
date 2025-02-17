from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self

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
    spelling: ParameterDef[StringAttr]

    def __init__(self, spelling: str = ""):
        super().__init__(self._parameters_from_spelling(spelling))

    @classmethod
    def _parameters_from_spelling(
        cls, spelling: str
    ) -> tuple[IntAttr | NoneAttr, StringAttr]:
        """
        Returns the parameter list required to construct a register instance from the given spelling.
        """
        index = cls.abi_index_by_name().get(spelling)
        index_attr = NoneAttr() if index is None else IntAttr(index)
        return index_attr, StringAttr(spelling)

    @property
    def register_name(self) -> str:
        """Returns name if allocated, raises ValueError if not"""
        if not self.is_allocated:
            raise ValueError("Cannot get name for unallocated register")
        return self.spelling.data

    @property
    def is_allocated(self) -> bool:
        """Returns true if the register is allocated, otherwise false"""
        return bool(self.spelling.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        if parser.parse_optional_punctuation("<"):
            name = parser.parse_identifier()
            parser.parse_punctuation(">")
        else:
            name = ""
        return cls._parameters_from_spelling(name)

    def print_parameters(self, printer: Printer) -> None:
        if self.spelling.data:
            with printer.in_angle_brackets():
                printer.print_string(self.spelling.data)

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
    def infinite_register_name(cls, index: int) -> str:
        """
        Provide the spelling for a register at the given index in the "infinite"
        register set.
        """
        raise NotImplementedError()

    @classmethod
    def infinite_register(cls, index: int) -> Self:
        """
        Provide the register at the given index in the "infinite" register set.
        """
        spelling = cls.infinite_register_name(index)
        res = cls(spelling)
        assert isinstance(res.index, NoneAttr), (
            f"Invalid 'infinite' register name {spelling}"
        )
        return res
