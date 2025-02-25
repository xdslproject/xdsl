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

    def __init__(self, spelling: str = ""):
        params = self._parameters_from_spelling(spelling)
        if params is None:
            raise ValueError(
                f"Invalid register spelling {spelling} for class {type(self).__name__}"
            )
        super().__init__(params)

    @classmethod
    def _parameters_from_spelling(
        cls, spelling: str
    ) -> tuple[IntAttr | NoneAttr, StringAttr] | None:
        """
        Returns the parameter list required to construct a register instance from the given spelling.
        """
        if not spelling:
            return NoneAttr(), StringAttr(spelling)
        index = cls.abi_index_by_name().get(spelling)
        if index is None:
            # Try to decode as infinite register
            prefix = cls.infinite_register_prefix()
            if spelling.startswith(prefix):
                suffix = spelling[len(prefix) :]
                # infinite registers go from -1 to -inf
                try:
                    index = ~int(suffix)
                except ValueError:
                    return None
            else:
                return None

        index_attr = IntAttr(index)
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
            start_pos = parser.pos
            name = parser.parse_identifier()
            end_pos = parser.pos
            parser.parse_punctuation(">")
            params = cls._parameters_from_spelling(name)
            if params is None:
                parser.raise_error(
                    f"Invalid register spelling {name} for class {cls.__name__}",
                    at_position=start_pos,
                    end_position=end_pos,
                )
        else:
            params = (StringAttr(""), NoneAttr())

        return params

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
    def infinite_register_prefix(cls) -> str:
        """
        Provide the prefix for the spelling for a register at the given index in the
        "infinite" register set.
        For a prefix `x`, the spelling of the first infinite register will be `x0`.
        """
        raise NotImplementedError()

    @classmethod
    def infinite_register(cls, index: int) -> Self:
        """
        Provide the register at the given index in the "infinite" register set.
        """
        spelling = cls.infinite_register_prefix() + str(index)
        res = cls(spelling)
        assert isinstance(res.index, IntAttr)
        assert res.index.data < 0, (
            f"Invalid 'infinite' register name: {spelling} clashes with finite register set"
        )
        return res
