from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from typing import Generic

from typing_extensions import TypeVar

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
    Parts of the input are always passed around as spans, so we know where they
    originated.
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


TokenKindT = TypeVar("TokenKindT", bound=Enum)


@dataclass
class Token(Generic[TokenKindT]):
    kind: TokenKindT

    span: Span

    @property
    def text(self):
        """The text composing the token."""
        return self.span.text


@dataclass
class Lexer(Generic[TokenKindT], ABC):
    input: Input
    """Input that is currently being lexed."""

    pos: Position = field(init=False, default=0)
    """
    Current position in the input.
    The position can be out of bounds, in which case the lexer is in EOF state.
    """

    def _form_token(self, kind: TokenKindT, start_pos: Position) -> Token[TokenKindT]:
        """
        Return a token with the given kind, and the start position.
        """
        return Token(kind, Span(start_pos, self.pos, self.input))

    @abstractmethod
    def lex(self) -> Token[TokenKindT]:
        """
        Lex a token from the input, and returns it.
        """
        raise NotImplementedError()
