from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import IO, Any

from typing_extensions import TypeVar


@dataclass(eq=False, repr=False)
class BasePrinter:
    stream: IO[str] | None = field(default=None)
    indent_num_spaces: int = field(default=2, kw_only=True)
    _indent: int = field(default=0, init=False)
    _current_line: int = field(default=0, init=False)
    _current_column: int = field(default=0, init=False)

    _next_line_callback: list[Callable[[], None]] = field(
        default_factory=list[Callable[[], None]], init=False
    )

    def print_string(self, text: str, *, indent: int | None = None) -> None:
        """
        Prints a string to the printer's output.

        This function takes into account indentation level when
        printing new lines.
        If the indentation level is specified as 0, the string is printed as-is, if `None`
        then the `Printer` instance's indentation level is used.
        """

        num_newlines = text.count("\n")

        if not num_newlines:
            self._current_column += len(text)
            print(text, end="", file=self.stream)
            return

        indent = self._indent if indent is None else indent
        lines = text.split("\n")

        if indent == 0 and not self._next_line_callback:
            # No indent and no callback to print after the next newline, the text
            # can be printed directly.
            self._current_line += num_newlines
            self._current_column = len(lines[-1])
            print(text, end="", file=self.stream)
            return

        # Line and column information is not computed ahead of time
        # as indent-aware newline printing may use it as part of
        # callbacks.
        print(lines[0], end="", file=self.stream)
        self._current_column += len(lines[0])
        for line in lines[1:]:
            self._print_new_line(indent=indent)
            print(line, end="", file=self.stream)
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
    def delimited(self, start: str, end: str):
        self.print_string(start)
        yield
        self.print_string(end)

    def in_angle_brackets(self):
        return self.delimited("<", ">")

    def in_braces(self):
        return self.delimited("{", "}")

    def in_parens(self):
        return self.delimited("(", ")")

    def in_square_brackets(self):
        return self.delimited("[", "]")

    def _print_new_line(
        self, indent: int | None = None, print_message: bool = True
    ) -> None:
        indent = self._indent if indent is None else indent
        # Prints a newline, bypassing the `print_string` method
        print(file=self.stream)
        self._current_line += 1
        if print_message:
            for callback in self._next_line_callback:
                callback()
            self._next_line_callback = []
        num_spaces = indent * self.indent_num_spaces
        # Prints indentation, bypassing the `print_string` method
        print(" " * num_spaces, end="", file=self.stream)
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
