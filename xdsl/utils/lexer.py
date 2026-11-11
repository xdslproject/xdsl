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
class Span(Generic[ContentT]):
    """
    Parts of the input are always passed around as spans, so we know where they
    originated.

    Positions and lengths use the input's native units: Unicode code points for
    strings and bytes for byte inputs. Converting the contents does not change
    these coordinates.
    """

    start: Position
    """
    Inclusive, zero-based start position within the input.
    """
    end: Position
    """
    Exclusive, zero-based end position within the input.
    """
    input: Input[ContentT]
    """
    The input being operated on
    """

    line_offset: int = 0
    """
    A line offset to add to the line number when reporting a location.
    """

    def __len__(self) -> Offset:
        return self.len

    @property
    def len(self) -> Offset:
        return self.end - self.start

    @property
    def content(self) -> ContentT:
        """The source slice, without encoding or decoding."""
        return self.input.content[self.start : self.end]

    @property
    def text(self) -> str:
        """The source slice as text, decoding byte inputs with strict UTF-8."""
        content = self.content
        return content.decode("utf-8") if isinstance(content, bytes) else content

    @property
    def bytes(self) -> bytes:
        """The source slice as bytes, encoding string inputs with strict UTF-8."""
        content = self.content
        return content.encode("utf-8") if isinstance(content, str) else content

    def get_location(self) -> Location:
        line_start = self.input.get_start_of_line(self.start)
        line_index_in_source = (
            self.input.content.count(self.input.newline, 0, line_start) + 1
        )
        line_index = line_index_in_source + self.line_offset
        column_index = self.start - line_start + 1
        return Location(self.input.name, line_index, column_index)

    def print_with_context(self, msg: str | None = None) -> str:
        """
        Returns a string containing lines relevant to the span. The Span's contents
        are highlighted by up-carets beneath them (`^`). The message msg is printed
        along these.

        Byte inputs are decoded as UTF-8 with replacement for display only;
        the reported location still uses byte columns.
        """
        context_start = self.input.get_start_of_line(self.start)
        # For empty spans, show the line containing the insertion point
        last_position = max(self.start, self.end - 1)
        context_end = self.input.get_end_of_line(last_position)
        context: str | bytes = self.input.content[context_start:context_end]
        start = self.start - context_start
        end = self.end - context_start

        if isinstance(context, bytes):
            # Approximate character positions, including partial or invalid UTF-8.
            start = len(context[:start].decode("utf-8", "replace"))
            end = len(context[:end].decode("utf-8", "replace"))
            context = context.decode("utf-8", "replace")

        capture = StringIO()
        print(self.get_location(), file=capture)

        line_start_in_context = 0
        for line in context.split("\n"):
            display_line = line.removesuffix("\r")
            start_column = min(max(start - line_start_in_context, 0), len(display_line))
            end_column = min(max(end - line_start_in_context, 0), len(display_line))
            print(display_line, file=capture)
            print(
                " " * start_column + "^" * max(end_column - start_column, 1),
                file=capture,
            )
            if msg is not None:
                print(" " * start_column + msg, file=capture)
                msg = None
            line_start_in_context += len(line) + 1
        return capture.getvalue()

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.start}:{self.end}]({self.content!r})"


TokenKindT = TypeVar("TokenKindT", bound=Enum)


@dataclass
class Token(Generic[TokenKindT, ContentT]):
    kind: TokenKindT

    span: Span[ContentT]

    @property
    def content(self) -> ContentT:
        """The source slice, without encoding or decoding."""
        return self.span.content

    @property
    def text(self) -> str:
        """The token as text, decoding byte inputs with strict UTF-8."""
        return self.span.text

    @property
    def bytes(self) -> bytes:
        """The token as bytes, encoding string inputs with strict UTF-8."""
        return self.span.bytes


@dataclass
class Lexer(ABC, Generic[TokenKindT, ContentT]):
    input: Input[ContentT]
    """Input that is currently being lexed."""

    pos: Position = field(init=False, default=0)
    """
    Current zero-based position within the input.
    The position can be out of bounds, in which case the lexer is in EOF state.
    """

    def _form_token(
        self, kind: TokenKindT, start_pos: Position
    ) -> Token[TokenKindT, ContentT]:
        """
        Return a token with the given kind, and the start position.
        """
        return Token(kind, Span(start_pos, self.pos, self.input))

    @abstractmethod
    def lex(self) -> Token[TokenKindT, ContentT]:
        """
        Lex a token from the input, and returns it.
        """
        raise NotImplementedError()
