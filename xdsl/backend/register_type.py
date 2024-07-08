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
    spelling: ParameterDef[StringAttr]

    def __init__(self, spelling: str):
        super().__init__(self._parameters_from_spelling(spelling))

    @classmethod
    def _parameters_from_spelling(
        cls, spelling: str
    ) -> tuple[IntAttr | NoneAttr, StringAttr]:
        """
        Returns the parameter list required to construct a register instance from the given spelling.
        """
        index_attr = NoneAttr()
        index = cls.abi_index_by_name().get(spelling)
        if index is not None:
            index_attr = IntAttr(index)
        return index_attr, StringAttr(spelling)

    @classmethod
    @abstractmethod
    def unallocated(cls) -> Self:
        raise NotImplementedError()

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
    @abstractmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        raise NotImplementedError()

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
