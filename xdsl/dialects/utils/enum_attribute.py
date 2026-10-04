from __future__ import annotations

from typing import ClassVar, cast, get_args, get_origin

from typing_extensions import TypeVar

from xdsl.ir import Data
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.str_enum import StrEnum

EnumType = TypeVar("EnumType", bound=StrEnum)


class EnumAttribute(Data[EnumType]):
    """
    Helper for Enum Attributes. Takes a StrEnum type parameter, and defines
    parsing/printing automatically from its values, restricted to be parsable as
    identifiers.

    example:
    ```python
    class MyEnum(StrEnum):
        First = auto()
        Second = auto()

    class MyEnumAttribute(EnumAttribute[MyEnum], SpacedOpaqueSyntaxAttribute):
        name = "example.my_enum"
    ```
    To use this attribute suffices to have a textual representation
    of `example<my_enum first>` and ``example<my_enum second>``

    """

    enum_type: ClassVar[type[StrEnum]]

    def __init_subclass__(cls) -> None:
        """
        Extract and store the Enum type used by the subclass for use in
        parsing/printing.

        Subclass implementations are also constrained to keep implementations
        reasonable, unless more complex use cases appear.

        The constraint(s) are:
        - Only direct, specialized inheritance is allowed. That is, using a
        subclass of EnumAttribute as a base class is *not supported*.
        This simplifies type-hacking code and I don't see it being too
        restrictive anytime soon.
        """
        super().__init_subclass__()

        orig_bases = getattr(cls, "__orig_bases__")
        enumattr = next(b for b in orig_bases if get_origin(b) is EnumAttribute)
        enum_type = get_args(enumattr)[0]
        if isinstance(enum_type, TypeVar):
            raise TypeError("Only direct inheritance from EnumAttribute is allowed.")

        cls.enum_type = enum_type

    def print_parameter(self, printer: Printer) -> None:
        printer.print_identifier_or_string_literal(self.data.value)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> EnumType:
        return cast(EnumType, parser.parse_str_enum(cls.enum_type))
