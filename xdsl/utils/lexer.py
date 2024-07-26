from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from string import hexdigits
from typing import Literal, TypeAlias, TypeGuard, cast, overload

from xdsl.utils.exceptions import ParseError

Position = int
"""
A position in a file.
The position correspond to the character index in the file.
"""


@dataclass(frozen=True)
class Input:
    """
    Used to keep track of the input when parsing.
    """

    content: str = field(repr=False)
    name: str
    len: int = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "len", len(self.content))

    def __len__(self):
        return self.len

    def get_lines_containing(self, span: Span) -> tuple[list[str], int, int] | None:
        # A pointer to the start of the first line
        start = 0
        line_no = span.line_offset
        source = self.content
        while True:
            next_start = source.find("\n", start)
            line_no += 1
            # Handle eof
            if next_start == -1:
                if span.start > len(source):
                    return None
                return [source[start:]], start, line_no
            # As long as the next newline comes before the spans start we can continue
            if next_start < span.start:
                start = next_start + 1
                continue
            # If the whole span is on one line, we are good as well
            if next_start >= span.end:
                return [source[start:next_start]], start, line_no
            while next_start < span.end:
                next_start = source.find("\n", next_start + 1)
                if next_start == -1:
                    next_start = span.end
            return source[start:next_start].split("\n"), start, line_no

    def at(self, i: Position) -> str | None:
        if i >= self.len:
            return None
        return self.content[i]

    def slice(self, start: Position, end: Position) -> str | None:
        if end > self.len or start < 0:
            return None
        return self.content[start:end]


@dataclass(frozen=True)
class Span:
    """
    Parts of the input are always passed around as spans, so we know where they originated.
    """

    start: Position
    """
    Start of tokens location in source file, global byte offset in file
    """
    end: Position
    """
    End of tokens location in source file, global byte offset in file
    """
    input: Input
    """
    The input being operated on
    """

    line_offset: int = 0
    """
    A line offset, to just add to ht file number in input when printed.
    """

    def __len__(self):
        return self.len

    @property
    def len(self):
        return self.end - self.start

    @property
    def text(self):
        return self.input.content[self.start : self.end]

    def get_line_col(self) -> tuple[int, int]:
        info = self.input.get_lines_containing(self)
        if info is None:
            return -1, -1
        _lines, offset_of_first_line, line_no = info
        return line_no, self.start - offset_of_first_line

    def print_with_context(self, msg: str | None = None) -> str:
        """
        returns a string containing lines relevant to the span. The Span's contents
        are highlighted by up-carets beneath them (`^`). The message msg is printed
        along these.
        """
        info = self.input.get_lines_containing(self)
        if info is None:
            return f"Unknown location of span {msg}. Error: "
        lines, offset_of_first_line, line_no = info
        # Offset relative to the first line:
        offset = self.start - offset_of_first_line
        remaining_len = max(self.len, 1)
        capture = StringIO()
        print(f"{self.input.name}:{line_no}:{offset}", file=capture)
        for line in lines:
            print(line, file=capture)
            if remaining_len < 0:
                continue
            len_on_this_line = min(remaining_len, len(line) - offset)
            remaining_len -= len_on_this_line
            print(
                "{}{}".format(" " * offset, "^" * max(len_on_this_line, 1)),
                file=capture,
            )
            if msg is not None:
                print("{}{}".format(" " * offset, msg), file=capture)
                msg = None
            offset = 0
        if msg is not None:
            print(msg, file=capture)
        return capture.getvalue()

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.start}:{self.end}](text='{self.text}')"


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


@dataclass
class Token:
    class Kind(Enum):
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
        def get_punctuation_spelling_to_kind_dict() -> dict[str, Token.Kind]:
            return {
                "->": Token.Kind.ARROW,
                ":": Token.Kind.COLON,
                ",": Token.Kind.COMMA,
                "...": Token.Kind.ELLIPSIS,
                "=": Token.Kind.EQUAL,
                ">": Token.Kind.GREATER,
                "{": Token.Kind.L_BRACE,
                "(": Token.Kind.L_PAREN,
                "[": Token.Kind.L_SQUARE,
                "<": Token.Kind.LESS,
                "-": Token.Kind.MINUS,
                "+": Token.Kind.PLUS,
                "?": Token.Kind.QUESTION,
                "}": Token.Kind.R_BRACE,
                ")": Token.Kind.R_PAREN,
                "]": Token.Kind.R_SQUARE,
                "*": Token.Kind.STAR,
                "|": Token.Kind.VERTICAL_BAR,
                "{-#": Token.Kind.FILE_METADATA_BEGIN,
                "#-}": Token.Kind.FILE_METADATA_END,
            }

        def is_punctuation(self) -> bool:
            punctuation_dict = Token.Kind.get_punctuation_spelling_to_kind_dict()
            return self in punctuation_dict.values()

        @staticmethod
        def is_spelling_of_punctuation(
            spelling: str,
        ) -> TypeGuard[PunctuationSpelling]:
            punctuation_dict = Token.Kind.get_punctuation_spelling_to_kind_dict()
            return spelling in punctuation_dict.keys()

        @staticmethod
        def get_punctuation_kind_from_spelling(
            spelling: PunctuationSpelling,
        ) -> Token.Kind:
            assert Token.Kind.is_spelling_of_punctuation(spelling), (
                "Kind.get_punctuation_kind_from_spelling: spelling is not a "
                "valid punctuation spelling!"
            )
            return Token.Kind.get_punctuation_spelling_to_kind_dict()[spelling]

    kind: Kind

    span: Span

    @property
    def text(self):
        """The text composing the token."""
        return self.span.text

    def get_int_value(self):
        """
        Translate the token text into an integer value.
        This function will raise an exception if the token is not an integer
        literal.
        """
        if self.kind != Token.Kind.INTEGER_LIT:
            raise ValueError("Token is not an integer literal!")
        if self.text[:2] in ["0x", "0X"]:
            return int(self.text, 16)
        return int(self.text, 10)

    def get_float_value(self):
        """
        Translate the token text into a float value.
        This function will raise an exception if the token is not a float
        literal.
        """
        if self.kind != Token.Kind.FLOAT_LIT:
            raise ValueError("Token is not a float literal!")
        return float(self.text)

    def get_string_literal_value(self) -> str:
        """
        Translate the token text into a string literal value.
        This will remove the quotes around the string literal, and unescape
        the string.
        This function will raise an exception if the token is not a string literal.
        """
        if self.kind != Token.Kind.STRING_LIT:
            raise ValueError("Token is not a string literal!")
        return StringLiteral.from_span(self.span).string_contents


@dataclass
class Lexer:
    input: Input
    """Input that is currently being lexed."""

    pos: Position = field(init=False, default=0)
    """
    Current position in the input.
    The position can be out of bounds, in which case the lexer is in EOF state.
    """

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

    def _form_token(self, kind: Token.Kind, start_pos: Position) -> Token:
        """
        Return a token with the given kind, and the start position.
        """
        return Token(kind, Span(start_pos, self.pos, self.input))

    def lex(self) -> Token:
        """
        Lex a token from the input, and returns it.
        """
        # First, skip whitespaces
        self._consume_whitespace()

        start_pos = self.pos
        current_char = self._get_chars()

        # Handle end of file
        if current_char is None:
            return self._form_token(Token.Kind.EOF, start_pos)

        # bare identifier
        if current_char.isalpha() or current_char == "_":
            return self._lex_bare_identifier(start_pos)

        # single-char punctuation that are not part of a multi-char token
        single_char_punctuation = {
            ":": Token.Kind.COLON,
            ",": Token.Kind.COMMA,
            "(": Token.Kind.L_PAREN,
            ")": Token.Kind.R_PAREN,
            "}": Token.Kind.R_BRACE,
            "[": Token.Kind.L_SQUARE,
            "]": Token.Kind.R_SQUARE,
            "<": Token.Kind.LESS,
            ">": Token.Kind.GREATER,
            "=": Token.Kind.EQUAL,
            "+": Token.Kind.PLUS,
            "*": Token.Kind.STAR,
            "?": Token.Kind.QUESTION,
            "|": Token.Kind.VERTICAL_BAR,
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
            return self._form_token(Token.Kind.ELLIPSIS, start_pos)

        # '-' and '->'
        if current_char == "-":
            if self._peek_chars() == ">":
                self._consume_chars()
                return self._form_token(Token.Kind.ARROW, start_pos)
            return self._form_token(Token.Kind.MINUS, start_pos)

        # '{' and '{-#'
        if current_char == "{":
            if self._peek_chars(2) == "-#":
                self._consume_chars(2)
                return self._form_token(Token.Kind.FILE_METADATA_BEGIN, start_pos)
            return self._form_token(Token.Kind.L_BRACE, start_pos)

        # '#-}'
        if current_char == "#" and self._peek_chars(2) == "-}":
            self._consume_chars(2)
            return self._form_token(Token.Kind.FILE_METADATA_END, start_pos)

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

    def _lex_bare_identifier(self, start_pos: Position) -> Token:
        """
        Lex a bare identifier with the following grammar:
        `bare-id ::= (letter|[_]) (letter|digit|[_$.])*`

        The first character is expected to have already been parsed.
        """
        self._consume_regex(self.bare_identifier_suffix_regex)

        return self._form_token(Token.Kind.BARE_IDENT, start_pos)

    def _lex_at_ident(self, start_pos: Position) -> Token:
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
            return self._form_token(Token.Kind.AT_IDENT, token.span.start)

        # literal string case
        if current_char == '"':
            token = self._lex_string_literal(start_pos)
            return self._form_token(Token.Kind.AT_IDENT, token.span.start)

        raise ParseError(
            Span(start_pos, self.pos, self.input),
            "@ identifier expected to start with letter, '_', or '\"'.",
        )

    _suffix_id = re.compile(r"([0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*)")

    def _lex_prefixed_ident(self, start_pos: Position) -> Token:
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
        assert (
            self.pos != 0
        ), "First prefixed identifier character must have been parsed"
        first_char = self.input.at(self.pos - 1)
        if first_char == "#":
            kind = Token.Kind.HASH_IDENT
        elif first_char == "!":
            kind = Token.Kind.EXCLAMATION_IDENT
        elif first_char == "^":
            kind = Token.Kind.CARET_IDENT
        else:
            assert (
                first_char == "%"
            ), "First prefixed identifier character must have been parsed correctly"
            kind = Token.Kind.PERCENT_IDENT

        match = self._consume_regex(self._suffix_id)
        if match is None:
            raise ParseError(
                Span(start_pos, self.pos, self.input),
                "Expected suffix identifier after {first_char}",
            )

        return self._form_token(kind, start_pos)

    _unescaped_characters_regex = re.compile(r'[^"\\\n\v\f]*')

    def _lex_string_literal(self, start_pos: Position) -> Token:
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
                    return self._form_token(Token.Kind.BYTES_LIT, start_pos)
                else:
                    return self._form_token(Token.Kind.STRING_LIT, start_pos)

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

    def _lex_number(self, start_pos: Position) -> Token:
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
            return self._form_token(Token.Kind.INTEGER_LIT, start_pos)

        # Decimal case
        self._consume_regex(self._digits_star_regex)

        # Check if we are lexing a floating point
        match = self._consume_regex(self._fractional_suffix_regex)
        if match is not None:
            return self._form_token(Token.Kind.FLOAT_LIT, start_pos)
        return self._form_token(Token.Kind.INTEGER_LIT, start_pos)
