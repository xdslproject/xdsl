from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from typing import Generic, NamedTuple
from warnings import warn

from typing_extensions import TypeVar

Position = int
"""
A zero-based byte offset in a source buffer.
"""


class Location(NamedTuple):
    """
    A file location using one-based line numbers and byte columns.

    This follows LLVM's [byte-column convention](https://clang.llvm.org/docs/UsersManual.html#opt-fshow-column).
    MLIR obtains its source coordinates from [LLVM's SourceMgr](https://github.com/llvm/llvm-project/blob/main/mlir/lib/AsmParser/Lexer.cpp).
    """

    file: str
    line: int
    "1-index of line in file."
    col: int
    "1-index of byte in line."

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


@dataclass(frozen=True, init=False)
class Input:
    """
    Source name and bytes contents.
    """

    content: bytes = field(repr=False)
    name: str
    len: int = field(init=False, repr=False)

    def __init__(self, content: bytes | str, name: str):
        if isinstance(content, str):
            warn(
                "Passing str to Input is deprecated; pass UTF-8 bytes instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            content = content.encode("utf-8")
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "len", len(content))

    def __len__(self):
        return self.len

    def get_start_of_line(self, pos: Position) -> Position:
        """
        Returns the location of the last newline before `pos`, or 0 if there are no
        previous newlines.
        """
        # Returns -1 if not found, meaning on first line
        pos = self.content.rfind(b"\n", 0, pos)
        # If the result is not the beginning of the input, go past the matching newline
        return pos + 1

    def get_end_of_line(self, pos: Position) -> Position:
        """
        Returns the position of the first newline after `pos`, or the length of the
        file if there are no more newlines.
        """
        pos = self.content.find(b"\n", pos)
        # If the result is at the end of the input, return the length for correct slice
        # indexing
        return self.len if pos == -1 else pos

    def at(self, i: Position) -> int | None:
        if i >= self.len:
            return None
        return self.content[i]

    def slice(self, start: Position, end: Position) -> bytes | None:
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
    Inclusive start of the span, as a byte offset in the source buffer.
    """
    end: Position
    """
    Exclusive end of the span, as a byte offset in the source buffer.
    """
    input: Input
    """
    The input being operated on
    """

    line_offset: int = 0
    """
    A line offset to add to the line number when reporting a location.
    """

    def __len__(self):
        return self.len

    @property
    def len(self):
        return self.end - self.start

    @property
    def bytes(self) -> bytes:
        return self.input.content[self.start : self.end]

    @property
    def text(self) -> str:
        """The source bytes decoded as UTF-8."""
        return self.bytes.decode("utf-8")

    def get_location(self) -> Location:
        line_start = self.input.get_start_of_line(self.start)
        line_index_in_source = self.input.content.count(b"\n", 0, line_start) + 1
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
        lines_start = self.input.get_start_of_line(self.start)
        lines_end = self.input.get_end_of_line(max(self.start, self.end - 1))
        lines = self.input.content[lines_start:lines_end].split(b"\n")
        capture = StringIO()
        print(loc, file=capture)
        line_start = lines_start
        for line in lines:
            source_line = line.removesuffix(b"\r")
            start = max(self.start - line_start, 0)
            end = max(self.end - line_start, 0)
            # Decode prefixes to translate byte offsets into character columns.
            # Replacement characters keep malformed UTF-8 printable.
            start_column = len(source_line[:start].decode("utf-8", "replace"))
            end_column = len(source_line[:end].decode("utf-8", "replace"))
            print(source_line.decode("utf-8", "replace"), file=capture)
            print(
                " " * start_column + "^" * max(end_column - start_column, 1),
                file=capture,
            )
            if msg is not None:
                print(" " * start_column + msg, file=capture)
                msg = None
            line_start += len(line) + 1
        return capture.getvalue()

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.start}:{self.end}]({self.bytes!r})"


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
