import re
from enum import Enum, auto
from string import hexdigits
from typing import TypeAlias, cast

from xdsl.utils.exceptions import ParseError
from xdsl.utils.lexer import Lexer, Position, Span, Token


class ToyTokenKind(Enum):
    SEMICOLON = auto()
    PARENTHESE_OPEN = auto()
    PARENTHESE_CLOSE = auto()
    BRACKET_OPEN = auto()
    BRACKET_CLOSE = auto()
    SBRACKET_OPEN = auto()
    SBRACKET_CLOSE = auto()
    LT = auto()
    GT = auto()
    EQ = auto()
    COMMA = auto()

    EOF = auto()

    RETURN = auto()
    VAR = auto()
    DEF = auto()

    IDENTIFIER = auto()
    NUMBER = auto()
    OPERATOR = auto()


SINGLE_CHAR_TOKENS = {
    ";": ToyTokenKind.SEMICOLON,
    "(": ToyTokenKind.PARENTHESE_OPEN,
    ")": ToyTokenKind.PARENTHESE_CLOSE,
    "{": ToyTokenKind.BRACKET_OPEN,
    "}": ToyTokenKind.BRACKET_CLOSE,
    "[": ToyTokenKind.SBRACKET_OPEN,
    "]": ToyTokenKind.SBRACKET_CLOSE,
    "<": ToyTokenKind.LT,
    ">": ToyTokenKind.GT,
    "=": ToyTokenKind.EQ,
    ",": ToyTokenKind.COMMA,
    "+": ToyTokenKind.OPERATOR,
    "-": ToyTokenKind.OPERATOR,
    "*": ToyTokenKind.OPERATOR,
    "/": ToyTokenKind.OPERATOR,
}

KEYWORD_TOKENS = {
    "return": ToyTokenKind.RETURN,
    "def": ToyTokenKind.DEF,
    "var": ToyTokenKind.VAR,
}

IDENTIFIER_CHARS = re.compile(r"[\w]|[\d]|_")
SPECIAL_CHARS = set(",")


ToyToken: TypeAlias = Token[ToyTokenKind]


class ToyLexer(Lexer[ToyTokenKind]):
    def _is_in_bounds(self, size: Position = 1) -> bool:
        """
        Check if the current position is within the bounds of the input.
        """
        return self.pos + size - 1 < self.input.len

    def _get_chars(self, size: int = 1) -> str | None:
        """
        Get the character at the current location, or multiple characters ahead.
        Return None if the position is out of bounds.
        """
        res = self.input.slice(self.pos, self.pos + size)
        self.pos += size
        return res

    def _peek_chars(self, size: int = 1) -> str | None:
        """
        Peek at the character at the current location, or multiple characters ahead.
        Return None if the position is out of bounds.
        """
        return self.input.slice(self.pos, self.pos + size)

    def _consume_chars(self, size: int = 1) -> None:
        """
        Advance the lexer position in the input by the given amount.
        """
        self.pos += size

    def _consume_regex(self, regex: re.Pattern[str]) -> re.Match[str] | None:
        """
        Advance the lexer position to the end of the next match of the given
        regular expression.
        """
        match = regex.match(self.input.content, self.pos)
        if match is None:
            return None
        self.pos = match.end()
        return match

    _whitespace_regex = re.compile(r"((#[^\n]*(\n)?)|(\s+))*", re.ASCII)

    def _consume_whitespace(self) -> None:
        """
        Consume whitespace and comments.
        """
        self._consume_regex(self._whitespace_regex)

    def lex(self) -> ToyToken:
        # First, skip whitespaces
        self._consume_whitespace()

        start_pos = self.pos
        current_char = self._get_chars()

        # Handle end of file
        if current_char is None:
            return self._form_token(ToyTokenKind.EOF, start_pos)

        # bare identifier
        if current_char.isalpha() or current_char == "_":
            return self._lex_bare_identifier(start_pos)

        # single-char punctuation that are not part of a multi-char token
        single_char_token_kind = SINGLE_CHAR_TOKENS.get(current_char)
        if single_char_token_kind is not None:
            return self._form_token(single_char_token_kind, start_pos)

        if current_char.isnumeric():
            return self._lex_number(start_pos)

        raise ParseError(
            Span(start_pos, start_pos + 1, self.input),
            f"Unexpected character: {current_char}",
        )

    IDENTIFIER_SUFFIX = r"[a-zA-Z0-9_$.]*"
    bare_identifier_suffix_regex = re.compile(IDENTIFIER_SUFFIX)

    def _lex_bare_identifier(self, start_pos: Position) -> ToyToken:
        """
        Lex a bare identifier with the following grammar:
        `bare-id ::= (letter|[_]) (letter|digit|[_$.])*`

        The first character is expected to have already been parsed.
        """
        self._consume_regex(self.bare_identifier_suffix_regex)

        span = Span(start_pos, self.pos, self.input)
        kind = KEYWORD_TOKENS.get(span.text, ToyTokenKind.IDENTIFIER)
        return Token(kind, span)

    _hexdigits_star_regex = re.compile(r"[0-9a-fA-F]*")
    _digits_star_regex = re.compile(r"[0-9]*")
    _fractional_suffix_regex = re.compile(r"\.[0-9]*([eE][+-]?[0-9]+)?")

    def _lex_number(self, start_pos: Position) -> ToyToken:
        """
        Lex a number literal, which is either a decimal or an hexadecimal.
        The first character is expected to have already been parsed.
        """
        first_digit = self.input.at(self.pos - 1)

        # Hexadecimal case, we only parse it if we see the first '0x' characters,
        # and then a first digit.
        # Otherwise, a string like '0xi32' would not be parsed correctly.
        if (
            first_digit == "0"
            and self._peek_chars() == "x"
            and self._is_in_bounds(2)
            and cast(str, self.input.at(self.pos + 1)) in hexdigits
        ):
            self._consume_chars(2)
            self._consume_regex(self._hexdigits_star_regex)
            return self._form_token(ToyTokenKind.NUMBER, start_pos)

        # Decimal case
        self._consume_regex(self._digits_star_regex)

        # Check if we are lexing a floating point
        match = self._consume_regex(self._fractional_suffix_regex)
        if match is not None:
            return self._form_token(ToyTokenKind.NUMBER, start_pos)
        return self._form_token(ToyTokenKind.NUMBER, start_pos)
