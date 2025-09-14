"""
This file contains the definition of `BaseParser`, a recursive descent parser
that is inherited from the different parsers used in xDSL.
"""

from dataclasses import dataclass
from typing import NoReturn

from typing_extensions import TypeVar

from xdsl.utils.lexer import Span
from xdsl.utils.mlir_lexer import MLIRTokenKind, PunctuationSpelling, StringLiteral
from xdsl.utils.str_enum import StrEnum

from .generic_parser import GenericParser  # noqa: TID251

_EnumType = TypeVar("_EnumType", bound=StrEnum)


@dataclass
class BaseParser(GenericParser[MLIRTokenKind]):
    def parse_optional_boolean(self) -> bool | None:
        """
        Parse a boolean, if present, with the format `true` or `false`.
        """
        if self._current_token.kind == MLIRTokenKind.BARE_IDENT:
            if self._current_token.text == "true":
                self._consume_token(MLIRTokenKind.BARE_IDENT)
                return True
            elif self._current_token.text == "false":
                self._consume_token(MLIRTokenKind.BARE_IDENT)
                return False
        return None

    def parse_boolean(self, context_msg: str = "") -> bool:
        """
        Parse a boolean with the format `true` or `false`.
        """
        return self.expect(
            lambda: self.parse_optional_boolean(),
            "Expected boolean literal" + context_msg,
        )

    def parse_optional_integer(
        self, allow_boolean: bool = True, allow_negative: bool = True
    ) -> int | None:
        """
        Parse an (possible negative) integer. The integer can either be
        decimal or hexadecimal.
        Optionally allow parsing of 'true' or 'false' into 1 and 0.
        """
        # Parse true and false if needed
        if allow_boolean:
            if (boolean := self.parse_optional_boolean()) is not None:
                return 1 if boolean else 0

        # Parse negative numbers if required
        is_negative = False
        if allow_negative:
            is_negative = self._parse_optional_token(MLIRTokenKind.MINUS) is not None

        # Parse the actual number
        if (int_token := self._parse_optional_token(MLIRTokenKind.INTEGER_LIT)) is None:
            if is_negative:
                self.raise_error("Expected integer literal after '-'")
            return None

        # Get the value and optionally negate it
        value = int_token.kind.get_int_value(int_token.span)
        if is_negative:
            value = -value
        return value

    def parse_integer(
        self,
        allow_boolean: bool = True,
        allow_negative: bool = True,
        context_msg: str = "",
    ) -> int:
        """
        Parse an (possible negative) integer. The integer can
        either be decimal or hexadecimal.
        Optionally allow parsing of 'true' or 'false' into 1 and 0.
        """

        return self.expect(
            lambda: self.parse_optional_integer(allow_boolean, allow_negative),
            "Expected integer literal" + context_msg,
        )

    def parse_optional_float(
        self,
        *,
        allow_negative: bool = True,
    ) -> float | None:
        """
        Parse a (possibly negative) float, if present.
        """
        is_negative = False
        if allow_negative:
            is_negative = self._parse_optional_token(MLIRTokenKind.MINUS) is not None

        if (value := self._parse_optional_token(MLIRTokenKind.FLOAT_LIT)) is not None:
            value = value.kind.get_float_value(value.span)
            return -value if is_negative else value

    def parse_float(
        self,
        *,
        allow_negative: bool = True,
    ) -> float:
        """
        Parse a (possibly negative) float.
        """

        return self.expect(
            lambda: self.parse_optional_float(allow_negative=allow_negative),
            "Expected float literal",
        )

    def parse_optional_number(
        self, *, allow_boolean: bool = False
    ) -> int | float | None:
        """
        Parse a (possibly negative) integer or float literal, if present.
        Can optionally parse 'true' or 'false' into 1 and 0.
        """

        is_negative = self._parse_optional_token(MLIRTokenKind.MINUS) is not None

        if (
            value := self.parse_optional_integer(
                allow_boolean=False, allow_negative=False
            )
        ) is not None:
            return -value if is_negative else value

        if (value := self.parse_optional_float(allow_negative=False)) is not None:
            return -value if is_negative else value

        if is_negative:
            self.raise_error("Expected integer or float literal after '-'")

        if allow_boolean and (value := self.parse_optional_boolean()) is not None:
            return 1 if value else 0

        return None

    def parse_number(
        self, allow_boolean: bool = False, context_msg: str = ""
    ) -> int | float:
        """
        Parse a (possibly negative) integer or float literal.
        Can optionally parse 'true' or 'false' into 1 and 0.
        """
        return self.expect(
            lambda: self.parse_optional_number(allow_boolean=allow_boolean),
            f"integer{', boolean,' if allow_boolean else ''} or float literal expected"
            + context_msg,
        )

    def parse_optional_str_literal(self) -> str | None:
        """
        Parse a string literal with the format `"..."`, if present.

        Returns the string contents without the quotes and with escape sequences
        resolved.
        """

        if (token := self._parse_optional_token(MLIRTokenKind.STRING_LIT)) is None:
            return None
        try:
            return token.kind.get_string_literal_value(token.span)
        except UnicodeDecodeError:
            return None

    def parse_str_literal(self, context_msg: str = "") -> str:
        """
        Parse a string literal with the format `"..."`.

        Returns the string contents without the quotes and with escape sequences
        resolved.
        """
        return self.expect(
            self.parse_optional_str_literal,
            "string literal expected" + context_msg,
        )

    def parse_optional_bytes_literal(self) -> bytes | None:
        """
        Parse a bytes literal with the format `"..."`, if present.

        Returns the bytes contents without the quotes and with escape sequences
        resolved.
        """

        if (token := self._parse_optional_token(MLIRTokenKind.BYTES_LIT)) is None:
            return None
        return StringLiteral.from_span(token.span).bytes_contents

    def parse_bytes_literal(self, context_msg: str = "") -> bytes:
        """
        Parse a bytes literal with the format `"..."`.

        Returns the bytes contents without the quotes and with escape sequences
        resolved.
        """
        return self.expect(
            self.parse_optional_bytes_literal,
            "bytes literal expected" + context_msg,
        )

    def parse_optional_identifier(self) -> str | None:
        """
        Parse an identifier, if present, with syntax:
            ident ::= (letter|[_]) (letter|digit|[_$.])*
        """
        if (token := self._parse_optional_token(MLIRTokenKind.BARE_IDENT)) is not None:
            return token.text
        return None

    def parse_identifier(self, context_msg: str = "") -> str:
        """
        Parse an identifier, if present, with syntax:
            ident ::= (letter|[_]) (letter|digit|[_$.])*
        """
        return self.expect(
            self.parse_optional_identifier, "identifier expected" + context_msg
        )

    def parse_optional_identifier_or_str_literal(self) -> str | None:
        """
        Parse an identifier or a string literal, if present.
            ident_or_str ::= ident | str_lit
        """

        if (ident := self.parse_optional_identifier()) is not None:
            return ident
        return self.parse_optional_str_literal()

    def parse_identifier_or_str_literal(self, context_msg: str = "") -> str:
        """
        Parse an identifier or a string literal, if present.
            ident_or_str ::= ident | str_lit
        """
        return self.expect(
            self.parse_optional_identifier_or_str_literal,
            "identifier or string literal expected" + context_msg,
        )

    def parse_optional_keyword(self, keyword: str) -> str | None:
        """Parse a specific identifier if it is present"""

        if (
            self._current_token.kind == MLIRTokenKind.BARE_IDENT
            and self._current_token.text == keyword
        ):
            self._consume_token(MLIRTokenKind.BARE_IDENT)
            return keyword
        return None

    def parse_keyword(self, keyword: str, context_msg: str = "") -> str:
        """Parse a specific identifier."""

        error_msg = f"Expected '{keyword}'" + context_msg
        if self.parse_optional_keyword(keyword) is not None:
            return keyword
        self.raise_error(error_msg)

    def parse_optional_punctuation(
        self, punctuation: PunctuationSpelling
    ) -> PunctuationSpelling | None:
        """
        Parse a punctuation, if it is present. Otherwise, return None.
        Punctuations are defined by `PunctuationSpelling`.
        """
        # This check is only necessary to catch errors made by users that
        # are not using pyright.
        assert MLIRTokenKind.is_spelling_of_punctuation(punctuation), (
            "'parse_optional_punctuation' must be called with a valid punctuation"
        )
        kind = MLIRTokenKind.get_punctuation_kind_from_name(punctuation)
        if self._parse_optional_token(kind) is not None:
            return punctuation
        return None

    def parse_punctuation(
        self, punctuation: PunctuationSpelling, context_msg: str = ""
    ) -> PunctuationSpelling:
        """
        Parse a punctuation. Punctuations are defined by
        `PunctuationSpelling`.
        """
        # This check is only necessary to catch errors made by users that
        # are not using pyright.
        assert MLIRTokenKind.is_spelling_of_punctuation(punctuation), (
            "'parse_punctuation' must be called with a valid punctuation"
        )
        kind = MLIRTokenKind.get_punctuation_kind_from_name(punctuation)
        self._parse_token(kind, f"Expected '{punctuation}'" + context_msg)
        return punctuation

    def _raise_wrong_str_enum_value_error(
        self, enum_type: type[_EnumType], span: Span
    ) -> NoReturn:
        """Raise an error for an unexpected string enum value."""
        enum_values = tuple(enum_type)
        if len(enum_values) == 1:
            self.raise_error(f"Expected `{enum_values[0]}`.", span)
        self.raise_error(
            f"Expected `{'`, `'.join(enum_values[:-1])}`, or `{enum_values[-1]}`.",
            span,
        )

    def parse_str_enum(self, enum_type: type[_EnumType]) -> _EnumType:
        """Parse a string enum value."""
        span = self._current_token.span
        result = self.parse_optional_str_enum(enum_type)
        if result is not None:
            return result
        self._raise_wrong_str_enum_value_error(enum_type, span)

    def parse_optional_str_enum(self, enum_type: type[_EnumType]) -> _EnumType | None:
        """Parse a string enum value, if present."""
        span = self._current_token.span
        value = self.parse_optional_identifier_or_str_literal()
        if value is None:
            return None
        if value not in enum_type.__members__.values():
            self._raise_wrong_str_enum_value_error(enum_type, span)
        return enum_type(value)
