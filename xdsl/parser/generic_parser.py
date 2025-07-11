"""
This file contains the definition of `BaseParser`, a recursive descent parser
that is inherited from the different parsers used in xDSL.
"""

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Generic, NoReturn, overload

from typing_extensions import TypeVar

from xdsl.utils.exceptions import ParseError
from xdsl.utils.lexer import Lexer, Position, Span, Token, TokenKindT


@dataclass(init=False)
class ParserState(Generic[TokenKindT]):
    """
    The parser state. It contains the lexer, and the next token to parse.
    The parser state should be shared between all parsers, so parsers can
    share the same position.
    """

    lexer: Lexer[TokenKindT]
    current_token: Token[TokenKindT]
    dialect_stack: list[str]

    def __init__(
        self, lexer: Lexer[TokenKindT], dialect_stack: list[str] | None = None
    ):
        if dialect_stack is None:
            dialect_stack = ["builtin"]
        self.lexer = lexer
        self.current_token = lexer.lex()
        self.dialect_stack = dialect_stack


_AnyInvT = TypeVar("_AnyInvT")


@dataclass
class GenericParser(Generic[TokenKindT]):
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

    _parser_state: ParserState[TokenKindT]

    @overload
    def raise_error(
        self,
        msg: str,
        at_position: Position,
        end_position: Position,
    ) -> NoReturn: ...

    @overload
    def raise_error(
        self,
        msg: str,
        at_position: Position | Span | None = None,
    ) -> NoReturn: ...

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

    def _resume_from(self, pos: Position | ParserState[TokenKindT]):
        """Resume parsing from a given position."""
        if isinstance(pos, Position):
            self.lexer.pos = pos
            self._parser_state.current_token = self.lexer.lex()
        else:
            self._parser_state = pos

    @property
    def _current_token(self) -> Token[TokenKindT]:
        """Get the token that is currently being parsed. Do not consume the token."""
        return self._parser_state.current_token

    @property
    def lexer(self) -> Lexer[TokenKindT]:
        """The lexer used to parse the current input."""
        return self._parser_state.lexer

    @property
    def pos(self) -> Position:
        """
        Get the starting position of the next token.
        This skips the whitespaces.
        """
        return self._current_token.span.start

    def _consume_token(
        self, expected_kind: TokenKindT | None = None
    ) -> Token[TokenKindT]:
        """
        Advance the lexer to the next token.
        Assert that the current token was of a specific kind, if specified.
        For reporting errors if the token was not of the expected kind,
        use `_parse_token` instead.
        """
        consumed_token = self._current_token
        assert expected_kind is None or consumed_token.kind == expected_kind, (
            f"Unexpected token {consumed_token}, expected {expected_kind}"
        )
        self._parser_state.current_token = self.lexer.lex()
        return consumed_token

    def _parse_optional_token(
        self, expected_kind: TokenKindT
    ) -> Token[TokenKindT] | None:
        """
        If the current token is of the expected kind, consume it and return it.
        Otherwise, return None.
        """
        if self._current_token.kind == expected_kind:
            return self._consume_token()

    def _parse_token(
        self, expected_kind: TokenKindT, error_msg: str
    ) -> Token[TokenKindT]:
        """
        Parse a specific token, and raise an error if it is not present.
        Returns the token that was parsed.
        """
        if (result := self._parse_optional_token(expected_kind)) is None:
            self.raise_error(error_msg, self._current_token.span)
        return result

    def _parse_optional_token_in(
        self, expected_kinds: Iterable[TokenKindT]
    ) -> Token[TokenKindT] | None:
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
        METADATA_TOKEN = ("{-#", "#-}")
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
                self.parse_characters(left_punctuation, context_msg)
                if self.parse_optional_characters(right_punctuation) is not None:
                    return []

        # Parse the list of elements
        elems = [parse()]
        while self.parse_optional_characters(",") is not None:
            elems.append(parse())

        # Parse the closing bracket, if a delimiter was provided
        match delimiter.value:
            case None:
                pass
            case (_, right_punctuation):
                self.parse_characters(right_punctuation, context_msg)

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
        if self.parse_optional_characters(left_punctuation) is None:
            return None
        if self.parse_optional_characters(right_punctuation) is not None:
            return []

        # Parse the list of elements
        elems = [parse()]
        while self.parse_optional_characters(",") is not None:
            elems.append(parse())

        # Parse the closing bracket
        self.parse_characters(right_punctuation, context_msg)

        return elems

    def parse_optional_undelimited_comma_separated_list(
        self,
        parse_optional: Callable[[], _AnyInvT | None],
        parse: Callable[[], _AnyInvT],
    ) -> list[_AnyInvT] | None:
        """
        Parses greedily a list of elements separated by commas, if a first element is
        present. The first element is parsed with `parse_optional`, and the remaining
        are parsed with `parse`. The parsing stops either if the first element is not
        present, or if no comma is present after parsing an element.
        """
        # Parse the first element, if it exist
        first_elem = parse_optional()
        if first_elem is None:
            return None

        # Parse the remaining elements
        elems = [first_elem]
        while self.parse_optional_characters(",") is not None:
            elems.append(parse())

        return elems

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

    @contextmanager
    def delimited(self, start: str, end: str):
        self.parse_characters(start)
        yield
        self.parse_characters(end)

    def in_angle_brackets(self):
        return self.delimited("<", ">")

    def in_square_brackets(self):
        return self.delimited("[", "]")

    def in_parens(self):
        return self.delimited("(", ")")

    def in_braces(self):
        return self.delimited("{", "}")
