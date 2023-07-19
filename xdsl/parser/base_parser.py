"""
This file contains the definition of `BaseParser`, a recursive descent parser
that is inherited from the different parsers used in xDSL.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, NoReturn, TypeVar, overload

from xdsl.utils.exceptions import ParseError
from xdsl.utils.lexer import Lexer, Position, Span, Token


@dataclass(init=False)
class ParserState:
    """
    The parser state. It contains the lexer, and the next token to parse.
    The parser state should be shared between all parsers, so parsers can
    share the same position.
    """

    lexer: Lexer
    current_token: Token

    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.current_token = lexer.lex()


_AnyInvT = TypeVar("_AnyInvT")


@dataclass
class BaseParser:
    """
    Basic recursive descent parser. It contains parsing methods that are shared
    between the different parsers.

    Methods named `parse_*` will consume tokens, and throw a `ParseError` if
    an unexpected token is parsed. Methods marked with `parse_optional` will return
    None if the first token is unexpected, and will throw a `ParseError` if the
    first token is expected, but a following token is not.

    Methods with a `context_msg` argument allows to append the context message to the
    thrown error. For instance, if `',' expected` is returned, setting `context_msg` to
    `" in integer list"` will throw the error `',' expected in integer list` instead.
    """

    _parser_state: ParserState

    @overload
    def raise_error(
        self,
        msg: str,
        at_position: Position,
        end_position: Position,
    ) -> NoReturn:
        ...

    @overload
    def raise_error(
        self,
        msg: str,
        at_position: Position | Span | None = None,
    ) -> NoReturn:
        ...

    def raise_error(
        self,
        msg: str,
        at_position: Span | Position | None = None,
        end_position: Position | None = None,
    ) -> NoReturn:
        """
        Raise a parsing exception at a specific location.
        If no location is provided, the current parser state location is used.
        """
        if end_position is not None:
            assert isinstance(at_position, Position)
            at_position = Span(at_position, end_position, self.lexer.input)
        if at_position is None:
            at_position = self._current_token.span
        elif isinstance(at_position, Position):
            at_position = Span(at_position, at_position, self.lexer.input)

        raise ParseError(at_position, msg)

    def _resume_from(self, pos: Position | ParserState):
        """Resume parsing from a given position."""
        if isinstance(pos, Position):
            self.lexer.pos = pos
            self._parser_state.current_token = self.lexer.lex()
        else:
            self._parser_state = pos

    @property
    def _current_token(self) -> Token:
        """Get the token that is currently being parsed. Do not consume the token."""
        return self._parser_state.current_token

    @property
    def lexer(self) -> Lexer:
        """The lexer used to parse the current input."""
        return self._parser_state.lexer

    @property
    def pos(self) -> Position:
        """
        Get the starting position of the next token.
        This skips the whitespaces.
        """
        return self._current_token.span.start

    def _consume_token(self, expected_kind: Token.Kind | None = None) -> Token:
        """
        Advance the lexer to the next token.
        Additionally check that the current token was of a specific kind,
        and assert if it was not.
        For reporting errors if the token was not of the expected kind,
        use `_parse_token` instead.
        """
        consumed_token = self._current_token
        if expected_kind is not None:
            assert consumed_token.kind == expected_kind, "Consumed an unexpected token!"
        self._parser_state.current_token = self.lexer.lex()
        return consumed_token

    def _parse_optional_token(self, expected_kind: Token.Kind) -> Token | None:
        """
        If the current token is of the expected kind, consume it and return it.
        Otherwise, return None.
        """
        if self._current_token.kind == expected_kind:
            current_token = self._current_token
            self._consume_token(expected_kind)
            return current_token
        return None

    def _parse_token(self, expected_kind: Token.Kind, error_msg: str) -> Token:
        """
        Parse a specific token, and raise an error if it is not present.
        Returns the token that was parsed.
        """
        if self._current_token.kind != expected_kind:
            self.raise_error(error_msg, self._current_token.span)
        current_token = self._current_token
        self._consume_token(expected_kind)
        return current_token

    def _parse_optional_token_in(
        self, expected_kinds: Iterable[Token.Kind]
    ) -> Token | None:
        """Parse one of the expected tokens if present, and returns it."""
        if self._current_token.kind not in expected_kinds:
            return None
        return self._consume_token()

    def expect(
        self, try_parse: Callable[[], _AnyInvT | None], error_message: str
    ) -> _AnyInvT:
        """
        This method is used to convert a `parse_optional_*` to a `parse_*`.
        It will run the parsing function, and raise an error if `None` was returned.
        """
        res = try_parse()
        if res is None:
            self.raise_error(error_message)
        return res

    class Delimiter(Enum):
        """
        Supported delimiters when parsing lists.
        """

        PAREN = ("(", ")")
        ANGLE = ("<", ">")
        SQUARE = ("[", "]")
        BRACES = ("{", "}")
        NONE = None

    def parse_comma_separated_list(
        self, delimiter: Delimiter, parse: Callable[[], _AnyInvT], context_msg: str = ""
    ) -> list[_AnyInvT]:
        """
        Parses greedily a list of elements separated by commas, and delimited
        by the specified delimiter. The parsing stops when the delimiter is
        closed, or when an error is produced. If no delimiter is specified, at
        least one element is expected to be parsed.
        """
        # Parse the opening bracket, and possibly the closing bracket, if a delimiter
        # was provided
        match delimiter.value:
            case None:
                pass
            case (left_punctuation, right_punctuation):
                self.parse_punctuation(left_punctuation, context_msg)
                if self.parse_optional_punctuation(right_punctuation) is not None:
                    return []

        # Parse the list of elements
        elems = [parse()]
        while self._parse_optional_token(Token.Kind.COMMA) is not None:
            elems.append(parse())

        # Parse the closing bracket, if a delimiter was provided
        match delimiter.value:
            case None:
                pass
            case (_, right_punctuation):
                self.parse_punctuation(right_punctuation, context_msg)

        return elems

    def parse_optional_comma_separated_list(
        self, delimiter: Delimiter, parse: Callable[[], _AnyInvT], context_msg: str = ""
    ) -> list[_AnyInvT] | None:
        """
        Parses greedily a list of elements separated by commas, and delimited
        by the specified delimiter. If no opening delimiter was found, return None.
        The parsing stops when the delimiter is closed, or when an error is produced.
        The NONE delimiter is not allowed by this method, use
        `parse_optional_undelimited_comma_separated_list` instead.
        """

        if delimiter == self.Delimiter.NONE:
            raise ValueError(
                "Cannot use `Delimiter.NONE` with "
                "`parse_optional_comma_separated_list`. Use "
                "`parse_optional_undelimited_comma_separated_list` instead."
            )

        # Parse the opening bracket, and possibly the closing bracket
        left_punctuation, right_punctuation = delimiter.value
        if self.parse_optional_punctuation(left_punctuation) is None:
            return None
        if self.parse_optional_punctuation(right_punctuation) is not None:
            return []

        # Parse the list of elements
        elems = [parse()]
        while self._parse_optional_token(Token.Kind.COMMA) is not None:
            elems.append(parse())

        # Parse the closing bracket
        self.parse_punctuation(right_punctuation, context_msg)

        return elems

    def parse_optional_undelimited_comma_separated_list(
        self,
        parse_optional: Callable[[], _AnyInvT | None],
        parse: Callable[[], _AnyInvT],
    ) -> list[_AnyInvT] | None:
        """
        Parses greedily a list of elements separated by commas, and delimited
        by the specified delimiter. Return None if no elements were parsed.
        """
        # Parse the first element, if it exist
        first_elem = parse_optional()
        if first_elem is None:
            return None
        elems = parse_optional()

        # Parse the remaining elements
        elems = [first_elem]
        while self._parse_optional_token(Token.Kind.COMMA) is not None:
            elems.append(parse())

        return elems

    def parse_optional_boolean(self) -> bool | None:
        """
        Parse a boolean, if present, with the format `true` or `false`.
        """
        if self._current_token.kind == Token.Kind.BARE_IDENT:
            if self._current_token.text == "true":
                self._consume_token(Token.Kind.BARE_IDENT)
                return True
            elif self._current_token.text == "false":
                self._consume_token(Token.Kind.BARE_IDENT)
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
            is_negative = self._parse_optional_token(Token.Kind.MINUS) is not None

        # Parse the actual number
        if (int_token := self._parse_optional_token(Token.Kind.INTEGER_LIT)) is None:
            if is_negative:
                self.raise_error("Expected integer literal after '-'")
            return None

        # Get the value and optionally negate it
        value = int_token.get_int_value()
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

    def parse_optional_number(self) -> int | float | None:
        """Parse a (possibly negative) integer or float literal, if present."""

        is_negative = self._parse_optional_token(Token.Kind.MINUS) is not None

        if (
            value := self.parse_optional_integer(
                allow_boolean=False, allow_negative=False
            )
        ) is not None:
            return -value if is_negative else value

        if (value := self._parse_optional_token(Token.Kind.FLOAT_LIT)) is not None:
            value = value.get_float_value()
            return -value if is_negative else value

        if is_negative:
            self.raise_error("Expected integer or float literal after '-'")
        return None

    def parse_number(self, context_msg: str = "") -> int | float:
        """Parse a (possibly negative) integer or float literal."""
        return self.expect(
            lambda: self.parse_optional_number(),
            "integer or float literal expected" + context_msg,
        )

    def parse_optional_str_literal(self) -> str | None:
        """
        Parse a string literal with the format `"..."`, if present.

        Returns the string contents without the quotes and with escape sequences
        resolved.
        """

        if (token := self._parse_optional_token(Token.Kind.STRING_LIT)) is None:
            return None
        return token.get_string_literal_value()

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

    def parse_optional_identifier(self) -> str | None:
        """
        Parse an identifier, if present, with syntax:
            ident ::= (letter|[_]) (letter|digit|[_$.])*
        """
        if (token := self._parse_optional_token(Token.Kind.BARE_IDENT)) is not None:
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

    def parse_optional_characters(self, text: str) -> str | None:
        """
        Parse a given token text, if present.
        If the given text is the beginning of the next token, this will still
        return None.
        """
        if self._current_token.text == text:
            self._consume_token()
            return text
        return None

    def parse_characters(self, text: str, context_msg: str = "") -> str:
        """
        Parse a given token text.
        The context message is appended to the error message if the parsing fails.
        If the given text is the start of the next token, this will still raise
        an error.
        """
        if (res := self.parse_optional_characters(text)) is not None:
            return res
        self.raise_error(f"'{text}' expected" + context_msg)

    def parse_optional_keyword(self, keyword: str) -> str | None:
        """Parse a specific identifier if it is present"""

        if (
            self._current_token.kind == Token.Kind.BARE_IDENT
            and self._current_token.text == keyword
        ):
            self._consume_token(Token.Kind.BARE_IDENT)
            return keyword
        return None

    def parse_keyword(self, keyword: str, context_msg: str = "") -> str:
        """Parse a specific identifier."""

        error_msg = f"Expected '{keyword}'" + context_msg
        if self.parse_optional_keyword(keyword) is not None:
            return keyword
        self.raise_error(error_msg)

    def parse_optional_punctuation(
        self, punctuation: Token.PunctuationSpelling
    ) -> Token.PunctuationSpelling | None:
        """
        Parse a punctuation, if it is present. Otherwise, return None.
        Punctuations are defined by `Token.PunctuationSpelling`.
        """
        # This check is only necessary to catch errors made by users that
        # are not using pyright.
        assert Token.Kind.is_spelling_of_punctuation(punctuation), (
            "'parse_optional_punctuation' must be " "called with a valid punctuation"
        )
        kind = Token.Kind.get_punctuation_kind_from_spelling(punctuation)
        if self._parse_optional_token(kind) is not None:
            return punctuation
        return None

    def parse_punctuation(
        self, punctuation: Token.PunctuationSpelling, context_msg: str = ""
    ) -> Token.PunctuationSpelling:
        """
        Parse a punctuation. Punctuations are defined by
        `Token.PunctuationSpelling`.
        """
        # This check is only necessary to catch errors made by users that
        # are not using pyright.
        assert Token.Kind.is_spelling_of_punctuation(
            punctuation
        ), "'parse_punctuation' must be called with a valid punctuation"
        kind = Token.Kind.get_punctuation_kind_from_spelling(punctuation)
        self._parse_token(kind, f"Expected '{punctuation}'" + context_msg)
        return punctuation
