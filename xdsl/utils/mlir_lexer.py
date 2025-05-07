from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from string import hexdigits
from typing import Literal, TypeAlias, TypeGuard, cast, overload

from xdsl.utils.exceptions import ParseError
from xdsl.utils.lexer import Lexer, Position, Span, Token

PunctuationSpelling: TypeAlias = Literal[
    "->",
    ":",
    ",",
    "...",
    "=",
    ">",
    "{",
    "(",
    "[",
    "<",
    "-",
    "+",
    "?",
    "}",
    ")",
    "]",
    "*",
    "|",
    "{-#",
    "#-}",
]


@dataclass(frozen=True, repr=False)
class StringLiteral(Span):
    def __post_init__(self):
        if len(self) < 2 or self.text[0] != '"' or self.text[-1] != '"':
            raise ParseError(self, "Invalid string literal!")

    @overload
    @classmethod
    def from_span(cls, span: Span) -> StringLiteral: ...

    @overload
    @classmethod
    def from_span(cls, span: None) -> None: ...

    @classmethod
    def from_span(cls, span: Span | None) -> StringLiteral | None:
        """
        Convert a normal span into a StringLiteral, to facilitate parsing.

        If argument is None, returns None.
        """
        if span is None:
            return None
        return cls(span.start, span.end, span.input)

    @property
    def string_contents(self):
        return self.bytes_contents.decode()

    @property
    def bytes_contents(self) -> bytes:
        bytes_contents = bytearray()
        iter_string = iter(self.text[1:-1])
        for c0 in iter_string:
            if c0 != "\\":
                bytes_contents += c0.encode()
            else:
                c0 = next(iter_string)
                match c0:
                    case "n":
                        bytes_contents += b"\n"
                    case "t":
                        bytes_contents += b"\t"
                    case "\\":
                        bytes_contents += b"\\"
                    case '"':
                        bytes_contents += b'"'
                    case _:
                        c1 = next(iter_string)
                        bytes_contents += int(c0 + c1, 16).to_bytes(1, "big")

        return bytes(bytes_contents)


class MLIRTokenKind(Enum):
    # Markers
    EOF = object()

    # Identifiers
    BARE_IDENT = object()
    """bare-id ::= (letter|[_]) (letter|digit|[_$.])*"""
    AT_IDENT = object()  # @foo
    """at-ident ::= `@` (bare-id | string-literal)"""
    HASH_IDENT = object()  # #foo
    """hash-ident  ::= `#` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)"""
    PERCENT_IDENT = object()  # %foo
    """percent-ident  ::= `%` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)"""
    CARET_IDENT = object()  # ^foo
    """caret-ident  ::= `^` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)"""
    EXCLAMATION_IDENT = object()  # !foo
    """exclamation-ident  ::= `!` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)"""

    # Literals
    FLOAT_LIT = object()  # 1.0
    INTEGER_LIT = object()  # 1
    STRING_LIT = object()  # "foo"
    BYTES_LIT = object()  # "foo\00\00"

    # Punctuation
    ARROW = "->"
    COLON = ":"
    COMMA = ","
    ELLIPSIS = "..."
    EQUAL = "="
    GREATER = ">"
    L_BRACE = "{"
    L_PAREN = "("
    L_SQUARE = "["
    LESS = "<"
    MINUS = "-"
    PLUS = "+"
    QUESTION = "?"
    R_BRACE = "}"
    R_PAREN = ")"
    R_SQUARE = "]"
    STAR = "*"
    VERTICAL_BAR = "|"
    FILE_METADATA_BEGIN = "{-#"
    FILE_METADATA_END = "#-}"

    @staticmethod
    def get_punctuation_spelling_to_kind_dict() -> dict[str, MLIRTokenKind]:
        return {
            "->": MLIRTokenKind.ARROW,
            ":": MLIRTokenKind.COLON,
            ",": MLIRTokenKind.COMMA,
            "...": MLIRTokenKind.ELLIPSIS,
            "=": MLIRTokenKind.EQUAL,
            ">": MLIRTokenKind.GREATER,
            "{": MLIRTokenKind.L_BRACE,
            "(": MLIRTokenKind.L_PAREN,
            "[": MLIRTokenKind.L_SQUARE,
            "<": MLIRTokenKind.LESS,
            "-": MLIRTokenKind.MINUS,
            "+": MLIRTokenKind.PLUS,
            "?": MLIRTokenKind.QUESTION,
            "}": MLIRTokenKind.R_BRACE,
            ")": MLIRTokenKind.R_PAREN,
            "]": MLIRTokenKind.R_SQUARE,
            "*": MLIRTokenKind.STAR,
            "|": MLIRTokenKind.VERTICAL_BAR,
            "{-#": MLIRTokenKind.FILE_METADATA_BEGIN,
            "#-}": MLIRTokenKind.FILE_METADATA_END,
        }

    def is_punctuation(self) -> bool:
        punctuation_dict = MLIRTokenKind.get_punctuation_spelling_to_kind_dict()
        return self in punctuation_dict.values()

    @staticmethod
    def is_spelling_of_punctuation(
        spelling: str,
    ) -> TypeGuard[PunctuationSpelling]:
        punctuation_dict = MLIRTokenKind.get_punctuation_spelling_to_kind_dict()
        return spelling in punctuation_dict.keys()

    @staticmethod
    def get_punctuation_kind_from_name(
        spelling: PunctuationSpelling,
    ) -> MLIRTokenKind:
        assert MLIRTokenKind.is_spelling_of_punctuation(spelling), (
            "Kind.get_punctuation_kind_from_name: spelling is not a "
            "valid punctuation spelling!"
        )
        return MLIRTokenKind.get_punctuation_spelling_to_kind_dict()[spelling]

    def get_int_value(self, span: Span):
        """
        Translate the token text into an integer value.
        This function will raise an exception if the token is not an integer
        literal.
        """
        if self != MLIRTokenKind.INTEGER_LIT:
            raise ValueError("Token is not an integer literal!")
        if span.text[:2] in ["0x", "0X"]:
            return int(span.text, 16)
        return int(span.text, 10)

    def get_float_value(self, span: Span):
        """
        Translate the token text into a float value.
        This function will raise an exception if the token is not a float
        literal.
        """
        if self != MLIRTokenKind.FLOAT_LIT:
            raise ValueError("Token is not a float literal!")
        return float(span.text)

    def get_string_literal_value(self, span: Span) -> str:
        """
        Translate the token text into a string literal value.
        This will remove the quotes around the string literal, and unescape
        the string.
        This function will raise an exception if the token is not a string literal.
        """
        if self != MLIRTokenKind.STRING_LIT:
            raise ValueError("Token is not a string literal!")
        return StringLiteral.from_span(span).string_contents


MLIRToken = Token[MLIRTokenKind]


@dataclass
class MLIRLexer(Lexer[MLIRTokenKind]):
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

    _whitespace_regex = re.compile(r"((//[^\n]*(\n)?)|(\s+))*", re.ASCII)

    def _consume_whitespace(self) -> None:
        """
        Consume whitespace and comments.
        """
        self._consume_regex(self._whitespace_regex)

    def _form_token(self, kind: MLIRTokenKind, start_pos: Position) -> MLIRToken:
        """
        Return a token with the given kind, and the start position.
        """
        return MLIRToken(kind, Span(start_pos, self.pos, self.input))

    def lex(self) -> MLIRToken:
        """
        Lex a token from the input, and returns it.
        """
        # First, skip whitespaces
        self._consume_whitespace()

        start_pos = self.pos
        current_char = self._get_chars()

        # Handle end of file
        if current_char is None:
            return self._form_token(MLIRTokenKind.EOF, start_pos)

        # bare identifier
        if current_char.isalpha() or current_char == "_":
            return self._lex_bare_identifier(start_pos)

        # single-char punctuation that are not part of a multi-char token
        single_char_punctuation = {
            ":": MLIRTokenKind.COLON,
            ",": MLIRTokenKind.COMMA,
            "(": MLIRTokenKind.L_PAREN,
            ")": MLIRTokenKind.R_PAREN,
            "}": MLIRTokenKind.R_BRACE,
            "[": MLIRTokenKind.L_SQUARE,
            "]": MLIRTokenKind.R_SQUARE,
            "<": MLIRTokenKind.LESS,
            ">": MLIRTokenKind.GREATER,
            "=": MLIRTokenKind.EQUAL,
            "+": MLIRTokenKind.PLUS,
            "*": MLIRTokenKind.STAR,
            "?": MLIRTokenKind.QUESTION,
            "|": MLIRTokenKind.VERTICAL_BAR,
        }
        if current_char in single_char_punctuation:
            return self._form_token(single_char_punctuation[current_char], start_pos)

        # '...'
        if current_char == ".":
            if self._get_chars(2) != "..":
                raise ParseError(
                    Span(start_pos, start_pos + 1, self.input),
                    "Expected three consecutive '.' for an ellipsis",
                )
            return self._form_token(MLIRTokenKind.ELLIPSIS, start_pos)

        # '-' and '->'
        if current_char == "-":
            if self._peek_chars() == ">":
                self._consume_chars()
                return self._form_token(MLIRTokenKind.ARROW, start_pos)
            return self._form_token(MLIRTokenKind.MINUS, start_pos)

        # '{' and '{-#'
        if current_char == "{":
            if self._peek_chars(2) == "-#":
                self._consume_chars(2)
                return self._form_token(MLIRTokenKind.FILE_METADATA_BEGIN, start_pos)
            return self._form_token(MLIRTokenKind.L_BRACE, start_pos)

        # '#-}'
        if current_char == "#" and self._peek_chars(2) == "-}":
            self._consume_chars(2)
            return self._form_token(MLIRTokenKind.FILE_METADATA_END, start_pos)

        # '@' and at-identifier
        if current_char == "@":
            return self._lex_at_ident(start_pos)

        # '#', '!', '^', '%' identifiers
        if current_char in ["#", "!", "^", "%"]:
            return self._lex_prefixed_ident(start_pos)

        if current_char == '"':
            return self._lex_string_literal(start_pos)

        if current_char.isnumeric():
            return self._lex_number(start_pos)

        raise ParseError(
            Span(start_pos, start_pos + 1, self.input),
            f"Unexpected character: {current_char}",
        )

    IDENTIFIER_SUFFIX = r"[a-zA-Z0-9_$.]*"
    bare_identifier_regex = re.compile(r"[a-zA-Z_]" + IDENTIFIER_SUFFIX)
    bare_identifier_suffix_regex = re.compile(IDENTIFIER_SUFFIX)

    def _lex_bare_identifier(self, start_pos: Position) -> MLIRToken:
        """
        Lex a bare identifier with the following grammar:
        `bare-id ::= (letter|[_]) (letter|digit|[_$.])*`

        The first character is expected to have already been parsed.
        """
        self._consume_regex(self.bare_identifier_suffix_regex)

        return self._form_token(MLIRTokenKind.BARE_IDENT, start_pos)

    def _lex_at_ident(self, start_pos: Position) -> MLIRToken:
        """
        Lex an at-identifier with the following grammar:
        `at-id ::= `@` (bare-id | string-literal)`

        The first character `@` is expected to have already been parsed.
        """
        current_char = self._get_chars()

        if current_char is None:
            raise ParseError(
                Span(start_pos, start_pos + 1, self.input),
                "Unexpected end of file after @.",
            )

        # bare identifier case
        if current_char.isalpha() or current_char == "_":
            token = self._lex_bare_identifier(start_pos)
            return self._form_token(MLIRTokenKind.AT_IDENT, token.span.start)

        # literal string case
        if current_char == '"':
            token = self._lex_string_literal(start_pos)
            return self._form_token(MLIRTokenKind.AT_IDENT, token.span.start)

        raise ParseError(
            Span(start_pos, self.pos, self.input),
            "@ identifier expected to start with letter, '_', or '\"'.",
        )

    _suffix_id = re.compile(r"([0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*)")

    def _lex_prefixed_ident(self, start_pos: Position) -> MLIRToken:
        """
        Parsed the following prefixed identifiers:
        ```
        hash-ident  ::= `#` suffix-id
        percent-ident  ::= `%` suffix-id
        caret-ident  ::= `^` suffix-id
        exclamation-ident  ::= `!` suffix-id
        ```
        with
        ```
        suffix-id = (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)
        ```

        The first character is expected to have already been parsed.
        """
        assert self.pos != 0, (
            "First prefixed identifier character must have been parsed"
        )
        first_char = self.input.at(self.pos - 1)
        if first_char == "#":
            kind = MLIRTokenKind.HASH_IDENT
        elif first_char == "!":
            kind = MLIRTokenKind.EXCLAMATION_IDENT
        elif first_char == "^":
            kind = MLIRTokenKind.CARET_IDENT
        else:
            assert first_char == "%", (
                "First prefixed identifier character must have been parsed correctly"
            )
            kind = MLIRTokenKind.PERCENT_IDENT

        match = self._consume_regex(self._suffix_id)
        if match is None:
            raise ParseError(
                Span(start_pos, self.pos, self.input),
                "Expected suffix identifier after {first_char}",
            )

        return self._form_token(kind, start_pos)

    _unescaped_characters_regex = re.compile(r'[^"\\\n\v\f]*')

    def _lex_string_literal(self, start_pos: Position) -> MLIRToken:
        """
        Lex a string literal.
        The first character `"` is expected to have already been parsed.
        """

        bytes_token = False
        while self._is_in_bounds():
            self._consume_regex(self._unescaped_characters_regex)
            current_char = self._get_chars()

            # end of string literal
            if current_char == '"':
                if bytes_token:
                    return self._form_token(MLIRTokenKind.BYTES_LIT, start_pos)
                else:
                    return self._form_token(MLIRTokenKind.STRING_LIT, start_pos)

            # newline character in string literal (not allowed)
            if current_char in ["\n", "\v", "\f"]:
                raise ParseError(
                    Span(start_pos, self.pos, self.input),
                    "Newline character not allowed in string literal.",
                )

            # escape character
            # TODO: handle unicode escape
            if current_char == "\\":
                escaped_char = self._get_chars()
                if escaped_char not in ['"', "\\", "n", "t"]:
                    bytes_token = True
                    next_char = self._get_chars()
                    if escaped_char is None or next_char is None:
                        raise ParseError(
                            Span(start_pos, self.pos, self.input),
                            "Unknown escape in string literal.",
                        )
                    try:
                        int(escaped_char + next_char, 16)
                    except Exception:
                        raise ParseError(
                            Span(start_pos, self.pos, self.input),
                            "Unknown escape in string literal.",
                        )

        raise ParseError(
            Span(start_pos, self.pos, self.input),
            "End of file reached before closing string literal.",
        )

    _hexdigits_star_regex = re.compile(r"[0-9a-fA-F]*")
    _digits_star_regex = re.compile(r"[0-9]*")
    _fractional_suffix_regex = re.compile(r"\.[0-9]*([eE][+-]?[0-9]+)?")

    def _lex_number(self, start_pos: Position) -> MLIRToken:
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
            return self._form_token(MLIRTokenKind.INTEGER_LIT, start_pos)

        # Decimal case
        self._consume_regex(self._digits_star_regex)

        # Check if we are lexing a floating point
        match = self._consume_regex(self._fractional_suffix_regex)
        if match is not None:
            return self._form_token(MLIRTokenKind.FLOAT_LIT, start_pos)
        return self._form_token(MLIRTokenKind.INTEGER_LIT, start_pos)
