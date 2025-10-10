import re
from dataclasses import dataclass
from typing import Generic

from typing_extensions import TypeVar

COMMENTS = r"(?:\/\/[^\n\r]+?(?:\*\)|[\n\r]))"
WHITESPACES = re.compile(r"(?:\s|" + COMMENTS + r")*")


@dataclass
class Location:
    pos: int


_LocatedCovT = TypeVar("_LocatedCovT", covariant=True)


@dataclass(frozen=True)
class Located(Generic[_LocatedCovT]):
    loc: Location
    value: _LocatedCovT

    def __bool__(self) -> bool:
        return bool(self.value)


class CodeCursor:
    code: str
    pos: int

    def __init__(self, code: str):
        self.code = code
        self.pos = 0

    def _whitespace_end(self) -> int:
        match = WHITESPACES.match(self.code, self.pos)
        assert match is not None
        return match.end()

    def skip_whitespaces(self):
        self.pos = self._whitespace_end()

    def next_regex(self, regex: re.Pattern[str]) -> Located[re.Match[str] | None]:
        match = self.peek_regex(regex)
        if match.value is not None:
            self.pos = match.value.end()
        return match

    def peek_regex(self, regex: re.Pattern[str]) -> Located[re.Match[str] | None]:
        pos = self._whitespace_end()
        return Located(Location(pos), regex.match(self.code, pos))


@dataclass
class ParseError(Exception):
    position: int
    msg: str

    @staticmethod
    def from_loc(loc: Location, msg: str) -> "ParseError":
        return ParseError(loc.pos, msg)

    def line_column(self, code: str) -> tuple[int, int]:
        if self.position < 0 or self.position > len(code):
            raise ValueError("error position is out of code range")

        line = code.count("\n", 0, self.position) + 1

        last_newline = code.rfind("\n", 0, self.position)
        column = self.position - last_newline
        return line, column
