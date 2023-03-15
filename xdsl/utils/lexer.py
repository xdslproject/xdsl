from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from enum import Enum, auto
from typing import Callable, cast
from string import hexdigits

from xdsl.utils.exceptions import ParseError


@dataclass(frozen=True)
class Input:
    """
    Used to keep track of the input when parsing.
    """
    content: str = field(repr=False)
    name: str

    @property
    def len(self):
        return len(self.content)

    def __len__(self):
        return self.len

    def get_lines_containing(self,
                             span: Span) -> tuple[list[str], int, int] | None:
        # A pointer to the start of the first line
        start = 0
        line_no = 0
        source = self.content
        while True:
            next_start = source.find('\n', start)
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
                next_start = source.find('\n', next_start + 1)
                if next_start == -1:
                    next_start = span.end
            return source[start:next_start].split('\n'), start, line_no

    def at(self, i: int) -> str | None:
        if i >= self.len:
            return None
        return self.content[i]

    def slice(self, start: int, end: int) -> str | None:
        if end > self.len or start < 0:
            return None
        return self.content[start:end]


@dataclass(frozen=True)
class Span:
    """
    Parts of the input are always passed around as spans, so we know where they originated.
    """

    start: int
    """
    Start of tokens location in source file, global byte offset in file
    """
    end: int
    """
    End of tokens location in source file, global byte offset in file
    """
    input: Input
    """
    The input being operated on
    """

    def __len__(self):
        return self.len

    @property
    def len(self):
        return self.end - self.start

    @property
    def text(self):
        return self.input.content[self.start:self.end]

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
            return "Unknown location of span {}. Error: ".format(msg)
        lines, offset_of_first_line, line_no = info
        # Offset relative to the first line:
        offset = self.start - offset_of_first_line
        remaining_len = max(self.len, 1)
        capture = StringIO()
        print("{}:{}:{}".format(self.input.name, line_no, offset),
              file=capture)
        for line in lines:
            print(line, file=capture)
            if remaining_len < 0:
                continue
            len_on_this_line = min(remaining_len, len(line) - offset)
            remaining_len -= len_on_this_line
            print("{}{}".format(" " * offset, "^" * max(len_on_this_line, 1)),
                  file=capture)
            if msg is not None:
                print("{}{}".format(" " * offset, msg), file=capture)
                msg = None
            offset = 0
        if msg is not None:
            print(msg, file=capture)
        return capture.getvalue()

    def __repr__(self):
        return "{}[{}:{}](text='{}')".format(self.__class__.__name__,
                                             self.start, self.end, self.text)


@dataclass
class Token:

    class Kind(Enum):
        # Markers
        EOF = auto()

        # Identifiers
        BARE_IDENT = auto()
        '''bare-id ::= (letter|[_]) (letter|digit|[_$.])*'''
        AT_IDENT = auto()  # @foo
        '''at-ident ::= `@` (bare-id | string-literal)'''
        HASH_IDENT = auto()  # #foo
        '''hash-ident  ::= `#` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)'''
        PERCENT_IDENT = auto()  # %foo
        '''percent-ident  ::= `%` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)'''
        CARET_IDENT = auto()  # ^foo
        '''caret-ident  ::= `^` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)'''
        EXCLAMATION_IDENT = auto()  # !foo
        '''exclamation-ident  ::= `!` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)'''

        # Literals
        FLOAT_LIT = auto()  # 1.0
        INTEGER_LIT = auto()  # 1
        STRING_LIT = auto()  # "foo"

        # Punctuation
        ARROW = auto()  # ->
        COLON = auto()  # :
        COMMA = auto()  # ,
        ELLIPSIS = auto()  # ...
        EQUAL = auto()  # =
        GREATER = auto()  # >
        L_BRACE = auto()  # {
        L_PAREN = auto()  # (
        L_SQUARE = auto()  # [
        LESS = auto()  # <
        MINUS = auto()  # -
        PLUS = auto()  # +
        QUESTION = auto()  # ?
        R_BRACE = auto()  # }
        R_PAREN = auto()  # )
        R_SQUARE = auto()  # ]
        STAR = auto()  # *
        VERTICAL_BAR = auto()  # |
        FILE_METADATA_BEGIN = auto()  # {-#
        FILE_METADATA_END = auto()  # #-}

    kind: Kind

    span: Span


@dataclass
class Lexer:
    input: Input
    """Input that is currently being lexed."""

    pos: int = field(init=False, default=0)
    """
    Current position in the input.
    The position can be out of bounds, in which case the lexer is in EOF state.
    """

    def _is_in_bounds(self, size: int = 1) -> bool:
        """
        Check if the current position is within the bounds of the input.
        """
        return self.pos + size - 1 < len(self.input.content)

    def _get_chars(self, size: int = 1) -> str | None:
        """
        Get the character at the current location, or multiple characters ahead.
        Return None if the position is out of bounds.
        """
        res = self.input.slice(self.pos, self.pos + size)
        self._consume_chars(size)
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
        self.pos = min(self.pos + size, len(self.input))

    def _consume_while(self, predicate: Callable[[str], bool]) -> None:
        """
        Advance the lexer position as long as the current character satisfies the
        given predicate.
        Returns the number of characters consumed.
        """
        while ((current := self._peek_chars()) is not None
               and predicate(current)):
            self._consume_chars()

    def _consume_whitespace(self) -> None:
        """
        Consume whitespace and comments.
        """
        while (current := self._peek_chars()) is not None:
            # Whitespace
            if current.isspace():
                self._consume_chars()
                continue

            # Comments
            if current == '/':
                if self._peek_chars(2) == '//':
                    self._consume_chars(2)
                    self._consume_while(lambda c: c != '\n')
                    continue

            return

    def _form_token(self, kind: Token.Kind, start_pos: int) -> Token:
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
        if current_char.isalpha() or current_char == '_':
            return self._lex_bare_identifier(start_pos)

        # single-char punctuation that are not part of a multi-char token
        single_char_punctuation = {
            ':': Token.Kind.COLON,
            ',': Token.Kind.COMMA,
            '(': Token.Kind.L_PAREN,
            ')': Token.Kind.R_PAREN,
            '}': Token.Kind.R_BRACE,
            '[': Token.Kind.L_SQUARE,
            ']': Token.Kind.R_SQUARE,
            '<': Token.Kind.LESS,
            '>': Token.Kind.GREATER,
            '=': Token.Kind.EQUAL,
            '+': Token.Kind.PLUS,
            '*': Token.Kind.STAR,
            '?': Token.Kind.QUESTION,
            '|': Token.Kind.VERTICAL_BAR
        }
        if current_char in single_char_punctuation:
            return self._form_token(single_char_punctuation[current_char],
                                    start_pos)

        # '...'
        if current_char == '.':
            if (self._get_chars(2) != '..'):
                raise ParseError(
                    Span(start_pos, start_pos + 1, self.input),
                    "Expected three consecutive '.' for an ellipsis",
                )
            return self._form_token(Token.Kind.ELLIPSIS, start_pos)

        # '-' and '->'
        if current_char == '-':
            if self._peek_chars() == '>':
                self._consume_chars()
                return self._form_token(Token.Kind.ARROW, start_pos)
            return self._form_token(Token.Kind.MINUS, start_pos)

        # '{' and '{-#'
        if current_char == '{':
            if (self._peek_chars(2) == '-#'):
                self._consume_chars(2)
                return self._form_token(Token.Kind.FILE_METADATA_BEGIN,
                                        start_pos)
            return self._form_token(Token.Kind.L_BRACE, start_pos)

        # '#-}'
        if (current_char == '#' and self._peek_chars(2) == '-}'):
            self._consume_chars(2)
            return self._form_token(Token.Kind.FILE_METADATA_END, start_pos)

        # '@' and at-identifier
        if current_char == '@':
            return self._lex_at_ident(start_pos)

        # '#', '!', '^', '%' identifiers
        if current_char in ['#', '!', '^', '%']:
            return self._lex_prefixed_ident(start_pos)

        if current_char == '"':
            return self._lex_string_literal(start_pos)

        if current_char.isnumeric():
            return self._lex_number(start_pos)

        raise ParseError(
            Span(start_pos, start_pos + 1, self.input),
            'Unexpected character: {}'.format(current_char),
        )

    def _lex_bare_identifier(self, start_pos: int) -> Token:
        """
        Lex a bare identifier with the following grammar:
        `bare-id ::= (letter|[_]) (letter|digit|[_$.])*`

        The first character is expected to have already been parsed.
        """
        self._consume_while(lambda c: c.isalnum() or c in ['_', '$', '.'])

        return self._form_token(Token.Kind.BARE_IDENT, start_pos)

    def _lex_at_ident(self, start_pos: int) -> Token:
        """
        Lex an at-identifier with the following grammar:
        `at-id ::= `@` (bare-id | string-literal)`

        The first character `@` is expected to have already been parsed.
        """
        current_char = self._get_chars()

        if current_char is None:
            raise ParseError(Span(start_pos, start_pos + 1, self.input),
                             "Unexpected end of file after @.")

        # bare identifier case
        if current_char.isalpha() or current_char == '_':
            token = self._lex_bare_identifier(start_pos)
            return self._form_token(Token.Kind.AT_IDENT, token.span.start)

        # literal string case
        if current_char == '"':
            token = self._lex_string_literal(start_pos)
            return self._form_token(Token.Kind.AT_IDENT, token.span.start)

        raise ParseError(
            Span(start_pos, self.pos, self.input),
            "@ identifier expected to start with letter, '_', or '\"'.")

    def _lex_prefixed_ident(self, start_pos: int) -> Token:
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
        suffix-id = (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)'''
        ```

        The first character is expected to have already been parsed.
        """
        assert self.pos != 0, "First prefixed identifier character must have been parsed"
        first_char = self.input.at(self.pos - 1)
        if first_char == '#':
            kind = Token.Kind.HASH_IDENT
        elif first_char == '!':
            kind = Token.Kind.EXCLAMATION_IDENT
        elif first_char == '^':
            kind = Token.Kind.CARET_IDENT
        elif first_char == '%':
            kind = Token.Kind.PERCENT_IDENT
        else:
            assert False, "First prefixed identifier character must have been parsed correctly"

        punctuation = ['$', '.', '_', '-']

        current_char = self._get_chars()
        if current_char is None:
            raise ParseError(
                Span(start_pos, start_pos + 1, self.input),
                "'first_char' is expected to follow with an identifier.")

        if current_char.isdigit():
            # digit+ case
            self._consume_while(lambda c: c.isdigit())
        elif current_char.isalpha() or current_char in punctuation:
            # (letter|[$._-]) (letter|[$._-]|digit)* case
            self._consume_while(lambda c: c.isalnum() or c in punctuation)
        else:
            raise ParseError(
                Span(start_pos, start_pos + 1, self.input),
                "Character expected to follow with an identifier.")

        return self._form_token(kind, start_pos)

    def _lex_string_literal(self, start_pos: int) -> Token:
        """
        Lex a string literal.
        The first character `"` is expected to have already been parsed.
        """

        while self._is_in_bounds():
            current_char = self._get_chars()

            # end of string literal
            if current_char == '"':
                return self._form_token(Token.Kind.STRING_LIT, start_pos)

            # newline character in string literal (not allowed)
            if current_char in ['\n', '\v', '\f']:
                raise ParseError(
                    Span(start_pos, self.pos, self.input),
                    "Newline character not allowed in string literal.")

            # escape character
            # TODO: handle unicode escape
            if current_char == '\\':
                escaped_char = self._get_chars()
                if escaped_char not in ['"', '\\', 'n', 't']:
                    raise ParseError(Span(self.pos - 1, self.pos, self.input),
                                     "Unknown escape in string literal.")

        raise ParseError(Span(start_pos, self.pos, self.input),
                         "End of file reached before closing string literal.")

    def _lex_number(self, start_pos: int) -> Token:
        """
        Lex a number literal, which is either a decimal or an hexadecimal.
        The first character is expected to have already been parsed.
        """
        first_digit = self.input.at(self.pos - 1)

        # Hexadecimal case, we only parse it if we see the first '0x' characters,
        # and then a first digit.
        # Otherwise, a string like '0xi32' would not be parsed correctly.
        if (first_digit == '0' and self._peek_chars() == 'x'
                and self._is_in_bounds(2)
                and cast(str, self.input.at(self.pos + 1)) in hexdigits):
            self._consume_chars(2)
            self._consume_while(lambda c: c in hexdigits)
            return self._form_token(Token.Kind.INTEGER_LIT, start_pos)

        # Decimal case
        self._consume_while(lambda c: c.isdigit())

        # Fractional part
        if self._peek_chars() != '.':
            return self._form_token(Token.Kind.INTEGER_LIT, start_pos)
        self._consume_chars()

        if not self._is_in_bounds():
            raise ParseError(Span(start_pos, self.pos, self.input),
                             "Decimal '.' expected to follow with a digit.")

        self._consume_while(lambda c: c.isdigit())

        # Exponent part
        if self._peek_chars() in ['e', 'E']:
            self._consume_chars()
            if not self._is_in_bounds():
                raise ParseError(
                    Span(start_pos, self.pos, self.input),
                    "Exponent expected to follow with a digit or a sign.")

            # Parse optionally a sign
            if self._peek_chars() in ['+', '-']:
                self._consume_chars()
                if not self._is_in_bounds():
                    raise ParseError(
                        Span(start_pos, self.pos, self.input),
                        "Sign expected to be followed with a digit.")

            self._consume_while(lambda c: c.isdigit())

        return self._form_token(Token.Kind.FLOAT_LIT, start_pos)
