from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from typing import Generic, NamedTuple

from typing_extensions import TypeVar

Position = int
"""
A position in a file.
The position correspond to the character index in the file.
"""


class Location(NamedTuple):
    "Structure definition a location in a file."

    file: str
    line: int
    "1-index of line in file"
    col: int
    "1-index of column in file"

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


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

    def get_start_of_line(self, pos: Position) -> Position:
        """
        Returns the location of the last newline before `pos`, or 0 if there are no
        previous newlines.
        """
        # Returns -1 if not found, meaning on first line
        pos = self.content.rfind("\n", 0, pos)
        # If the result is not the beginning of the input, go past the matching newline
        return pos + 1

    def get_end_of_line(self, pos: Position) -> Position:
        """
        Returns the position of the first newline after `pos`, or the length of the
        file if there are no more newlines.
        """
        pos = self.content.find("\n", pos)
        # If the result is at the end of the input, return the length for correct slice
        # indexing
        return self.len if pos == -1 else pos

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
