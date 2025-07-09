import re
from typing import NamedTuple

from typing_extensions import Any

from xdsl.utils.lexer import Token


class Location(NamedTuple):
    "Structure definition a location in a file."

    file: str
    line: int
    col: int

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


_NEWLINE = re.compile(r"\n")


def loc(token: Token[Any]) -> Location:
    file = token.span.input.name
    # Could be much faster

    remaining = token.span.start
    prev_end = 0

    for line, newline_match in enumerate(
        re.finditer(_NEWLINE, token.span.input.content)
    ):
        len_line = newline_match.start() - prev_end
        if remaining < len_line:
            return Location(file, line + 1, remaining + 1)
        remaining -= len_line + 1
        prev_end = newline_match.end()

    raise AssertionError(f"Could not find location of token {token}")
