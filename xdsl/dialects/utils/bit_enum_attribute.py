from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import ClassVar, Generic, cast, get_args, get_origin

from typing_extensions import TypeVar, deprecated

from xdsl.ir import Data
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.utils.str_enum import StrEnum

EnumType = TypeVar("EnumType", bound=StrEnum)


@dataclass(frozen=True, init=False)
class BitEnumAttribute(Data[frozenset[EnumType]], Generic[EnumType]):
    """
    Core helper for BitEnumAttributes. Takes a StrEnum type parameter, and
    defines parsing/printing automatically from its values.

    Additionally, two values can be given to designate all/none bits being set.

    example:
    ```python
    class MyBitEnum(StrEnum):
        First = auto()
        Second = auto()

    class MyBitEnumAttribute(BitEnumAttribute[MyBitEnum]):
        name = "example.my_bit_enum"
        none_value = "none"
        all_value = "all"

    """

    enum_type: ClassVar[type[StrEnum]]
    none_value: ClassVar[str | None] = None
    all_value: ClassVar[str | None] = None
    separator_value: ClassVar[str] = ","
    delimiter_value: ClassVar[Parser.Delimiter] = Parser.Delimiter.ANGLE

    def __init__(self, flags: None | Iterable[EnumType] | str) -> None:
        flags_: frozenset[EnumType]
        match flags:
            case self.none_value | None:
                flags_ = frozenset()
            case self.all_value:
                flags_ = cast(frozenset[EnumType], frozenset(self.enum_type))
            case other if isinstance(other, str):
                raise TypeError(
                    f"expected string parameter to be one of {self.none_value} or {self.all_value}, got {other}"
                )
            case other:
                assert not isinstance(other, str)
                flags_ = frozenset(other)

        super().__init__(flags_)

    def __init_subclass__(cls) -> None:
        """
        Extract and store the Enum type used by the subclass for use in
        parsing/printing.

        Subclass implementations are also constrained to keep implementations
        reasonable, unless more complex use cases appear.

        The constraint(s) are:
        - Only direct, specialized inheritance is allowed. That is, using a
        subclass of BitEnumAttribute as a base class is *not supported*.
        This simplifies type-hacking code and I don't see it being too
        restrictive anytime soon.
        """
        super().__init_subclass__()

        orig_bases = getattr(cls, "__orig_bases__")
        enumattr = next(b for b in orig_bases if get_origin(b) is BitEnumAttribute)
        enum_type = get_args(enumattr)[0]
        if isinstance(enum_type, TypeVar):
            raise TypeError("Only direct inheritance from BitEnumAttribute is allowed.")

        cls.enum_type = enum_type

    @property
    @deprecated("Please use .data instead")
    def flags(self) -> set[EnumType]:
        return set(self.data)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> frozenset[EnumType]:
        def parse_element() -> set[EnumType]:
            if (
                cls.none_value is not None
                and parser.parse_optional_keyword(cls.none_value) is not None
            ):
                return set()
            if (
                cls.all_value is not None
                and parser.parse_optional_keyword(cls.all_value) is not None
            ):
                return set(cast(Iterable[EnumType], cls.enum_type))
            value = parser.parse_str_enum(cls.enum_type)
            return {cast(type[EnumType], cls.enum_type)(value)}

        flag_sets = parser.parse_list(
            cls.delimiter_value, parse_element, cls.separator_value
        )

        if not flag_sets:
            return frozenset()

        res: set[EnumType] = set()

        for flag_set in flag_sets:
            res |= flag_set

        return frozenset(res)

    def print_parameter(self, printer: Printer):
        match self.delimiter_value:
            case Parser.Delimiter.NONE:
                delimiter = nullcontext()
            case _:
                delimiter = printer.delimited(*self.delimiter_value.value)

        with delimiter:
            flags = self.data
            if not flags and self.none_value is not None:
                printer.print_string(self.none_value)
            elif len(flags) == len(self.enum_type) and self.all_value is not None:
                printer.print_string(self.all_value)
            else:
                # make sure we emit flags in a consistent order
                printer.print_list(
                    tuple(flag.value for flag in self.enum_type if flag in flags),
                    printer.print_string,
                    self.separator_value,
                )
