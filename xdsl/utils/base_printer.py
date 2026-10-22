from __future__ import annotations

import sys
from collections.abc import Callable, Generator, Iterable
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from io import BufferedIOBase, RawIOBase, TextIOBase
from typing import IO, Any, Generic, cast, overload

from typing_extensions import TypeVar

from xdsl.utils.lexer import ContentT, Offset


@dataclass(eq=False, repr=False, init=False)
class BasePrinter(Generic[ContentT]):
    """
    Print native content to a text or binary stream, defaulting to stdout.

    Standard text streams use `str`, and standard binary streams use `str.encode`
    (UTF-8). Custom streams require an explicit `encode` function.
    """

    stream: IO[ContentT]
    """Output stream, defaulting to stdout at construction time."""

    newline: ContentT
    indent_num_spaces: int
    _indent: int
    _current_line: int
    _current_column: Offset
    _space: ContentT
    _encode: Callable[[str], ContentT]
    _next_line_callback: list[Callable[[], None]]

    @overload
    def __init__(
        self: BasePrinter[str],
        stream: IO[str] | None = None,
        encode: Callable[[str], str] | None = None,
        *,
        indent_num_spaces: int = 2,
    ) -> None: ...

    @overload
    def __init__(
        self,
        stream: IO[ContentT],
        encode: Callable[[str], ContentT] | None = None,
        *,
        indent_num_spaces: int = 2,
    ) -> None: ...

    def __init__(
        self,
        stream: IO[Any] | None = None,
        encode: Callable[[str], str | bytes] | None = None,
        *,
        indent_num_spaces: int = 2,
    ) -> None:
        if stream is None:
            stream = sys.stdout
            if encode is None:
                encode = str
        if encode is None:
            if isinstance(stream, TextIOBase):
                encode = str
            elif isinstance(stream, BufferedIOBase | RawIOBase):
                encode = str.encode
            else:
                raise TypeError("encode must be supplied for custom streams")
        self.stream = cast(IO[ContentT], stream)
        self._encode = cast(Callable[[str], ContentT], encode)
        self.indent_num_spaces = indent_num_spaces
        self._indent = 0
        self._current_line = 0
        self._current_column = 0
        self._next_line_callback = []
        self.newline = self._encode("\n")
        self._space = self._encode(" ")

    def _write(self, content: ContentT) -> None:
        self.stream.write(content)

    def print_string(self, text: str, *, indent: int | None = None) -> None:
        """Print Python text using the configured encoding function."""
        self.print_content(self._encode(text), indent=indent)

    def print_content(self, text: ContentT, *, indent: int | None = None) -> None:
        """
        Print native content without encoding or decoding.

        This function takes into account indentation level when
        printing new lines.
        If the indentation level is specified as 0, the string is printed as-is, if `None`
        then the `Printer` instance's indentation level is used.
        """

        num_newlines = text.count(self.newline)

        if not num_newlines:
            self._current_column += len(text)
            self._write(text)
            return

        indent = self._indent if indent is None else indent
        lines = cast(list[ContentT], text.split(self.newline))

        if indent == 0 and not self._next_line_callback:
            # No indent and no callback to print after the next newline, the text
            # can be printed directly.
            self._current_line += num_newlines
            self._current_column = len(lines[-1])
            self._write(text)
            return

        # Line and column information is not computed ahead of time
        # as indent-aware newline printing may use it as part of
        # callbacks.
        self._write(lines[0])
        self._current_column += len(lines[0])
        for line in lines[1:]:
            self._print_new_line(indent=indent)
            self._write(line)
            self._current_column += len(line)

    T = TypeVar("T")

    def print_list(
        self, elems: Iterable[T], print_fn: Callable[[T], Any], delimiter: str = ", "
    ) -> None:
        for i, elem in enumerate(elems):
            if i:
                self.print_string(delimiter)
            print_fn(elem)

    @contextmanager
    def delimited(self, start: str, end: str) -> Generator[None, None, None]:
        self.print_string(start)
        yield
        self.print_string(end)

    def in_angle_brackets(self) -> AbstractContextManager[None]:
        return self.delimited("<", ">")

    def in_braces(self) -> AbstractContextManager[None]:
        return self.delimited("{", "}")

    def in_parens(self) -> AbstractContextManager[None]:
        return self.delimited("(", ")")

    def in_square_brackets(self) -> AbstractContextManager[None]:
        return self.delimited("[", "]")

    def _print_new_line(
        self, indent: int | None = None, print_message: bool = True
    ) -> None:
        indent = self._indent if indent is None else indent
        # Prints a newline, bypassing the `print_string` method
        self._write(self.newline)
        self._current_line += 1
        if print_message:
            for callback in self._next_line_callback:
                callback()
            self._next_line_callback = []
        num_spaces = indent * self.indent_num_spaces
        # Prints indentation, bypassing the `print_string` method
        self._write(self._space * num_spaces)
        self._current_column = num_spaces

    @contextmanager
    def indented(self, amount: int = 1):
        """
        Increases the indentation level by the provided amount
        for the duration of the context.

        Only affects new lines printed within the context.
        """

        self._indent += amount
        try:
            yield
        finally:
            self._indent -= amount

    def _add_message_on_next_line(self, message: str, begin_pos: int, end_pos: int):
        """Add a message that will be displayed on the next line."""

        def callback(indent: int = self._indent):
            self._print_message(message, begin_pos, end_pos, indent)

        self._next_line_callback.append(callback)

    def _print_message(
        self, message: str, begin_pos: int, end_pos: int, indent: int | None = None
    ):
        """
        Print a message.
        This is expected to be called at the beginning of a new line and to create a new
        line at the end.
        The span of the message to be underlined is represented as [begin_pos, end_pos).
        """
        indent = self._indent if indent is None else indent
        indent_size = indent * self.indent_num_spaces
        self.print_string(" " * indent_size)
        message_end_pos = max(map(len, message.split("\n"))) + indent_size + 2
        first_line = (
            (begin_pos - indent_size) * "-"
            + (end_pos - begin_pos) * "^"
            + (max(message_end_pos, end_pos) - end_pos) * "-"
        )
        self.print_string(first_line)
        self._print_new_line(indent=indent, print_message=False)
        for message_line in message.split("\n"):
            self.print_string("| ")
            self.print_string(message_line)
            self._print_new_line(indent=indent, print_message=False)
        self.print_string("-" * (max(message_end_pos, end_pos) - indent_size))
        self._print_new_line(indent=0, print_message=False)
