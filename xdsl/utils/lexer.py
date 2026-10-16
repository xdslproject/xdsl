from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from typing import Generic, NamedTuple, overload

from typing_extensions import TypeVar

ContentT = TypeVar("ContentT", str, bytes, default=str)

Position = int
"""
Zero-based position from the start of the input, measured in Unicode code points
for `str` or bytes for `bytes`.
"""

Offset = int
"""
Signed displacement between positions, measured in Unicode code points for `str`
or bytes for `bytes`.
"""

Column = int
"""
One-based position within a line, measured in Unicode code points for `str`
or bytes for `bytes`.
"""


class Location(NamedTuple):
    """A file location using the originating input's coordinate units."""

    file: str
    line: int
    "One-based line number."
    col: Column
    "One-based position within the line."

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


@dataclass(frozen=True)
class Input(Generic[ContentT]):
    """
    Source name and contents, preserved without encoding or decoding.
    """

    content: ContentT = field(repr=False)
    name: str
    len: Offset = field(init=False, repr=False)
    newline: ContentT = field(init=False, repr=False, compare=False)
    """LF in the source representation."""

    def __post_init__(self):
        object.__setattr__(self, "len", len(self.content))
        object.__setattr__(
            self, "newline", "\n" if isinstance(self.content, str) else b"\n"
        )

    def __len__(self) -> Offset:
        return self.len

    def get_start_of_line(self, pos: Position) -> Position:
        """
        Returns the location of the last newline before `pos`, or 0 if there are no
        previous newlines.
        """
        # Returns -1 if not found, meaning on first line
        pos = self.content.rfind(self.newline, 0, pos)
        # If the result is not the beginning of the input, go past the matching newline
        return pos + 1

    def get_end_of_line(self, pos: Position) -> Position:
        """
        Returns the position of the first newline after `pos`, or the length of the
        file if there are no more newlines.
        """
        pos = self.content.find(self.newline, pos)
        # If the result is at the end of the input, return the length for correct slice
        # indexing
        return self.len if pos == -1 else pos

    @overload
    def at(self: Input[str], i: Position) -> str | None: ...

    @overload
    def at(self: Input[bytes], i: Position) -> int | None: ...

    @overload
    def at(self, i: Position) -> str | int | None: ...

    def at(self, i: Position) -> str | int | None:
        if i >= self.len:
            return None
        return self.content[i]

    def slice(self, start: Position, end: Position) -> ContentT | None:
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
    Inclusive, zero-based start position within the input.
    """
    end: Position
    """
    Exclusive, zero-based end position within the input.
    """
    input: Input[str]
    """
    The input being operated on
    """

    line_offset: int = 0
    """
    A line offset, to just add to ht file number in input when printed.
    """

    def __len__(self) -> Offset:
        return self.len

    @property
    def len(self) -> Offset:
        return self.end - self.start

    @property
    def text(self):
        return self.input.content[self.start : self.end]

    def get_location(self) -> Location:
        line_start = self.input.get_start_of_line(self.start)
        line_index_in_source = self.input.content.count("\n", 0, line_start) + 1
        line_index = line_index_in_source + self.line_offset
        column_index = self.start - line_start + 1
        return Location(self.input.name, line_index, column_index)

    def print_with_context(self, msg: str | None = None) -> str:
        """
        returns a string containing lines relevant to the span. The Span's contents
        are highlighted by up-carets beneath them (`^`). The message msg is printed
        along these.
        """
        loc = self.get_location()
        # Offset relative to the first line:
        offset = loc.col - 1
        lines_start = self.start - offset
        lines_end = self.input.get_end_of_line(self.end)
        lines = self.input.content[lines_start:lines_end].splitlines()
        remaining_len = max(self.len, 1)
        capture = StringIO()
        print(loc, file=capture)
        for line in lines:
            print(line, file=capture)
            if remaining_len < 0:
                continue
            caret_count = min(remaining_len, max(len(line) - offset, 1))
            print(" " * offset + "^" * caret_count, file=capture)
            if msg is not None:
                print(" " * offset + msg, file=capture)
                msg = None
            remaining_len -= caret_count
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
class Lexer(ABC, Generic[TokenKindT]):
    input: Input[str]
    """Input that is currently being lexed."""

    pos: Position = field(init=False, default=0)
    """
    Current zero-based position within the input.
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
