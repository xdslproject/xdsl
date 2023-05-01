import re

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .location import Location


@dataclass(init=False)
class Token:
    file: Path
    line: int
    col: int
    text: str

    def __init__(self, file: Path, line: int, col: int, text: str):
        self.file = file
        self.line = line
        self.col = col
        self.text = text

    @property
    def loc(self):
        return Location(self.file, self.line, self.col)

    @classmethod
    def name(cls):
        return cls.__name__


@dataclass
class IdentifierToken(Token):
    pass


@dataclass
class NumberToken(Token):
    value: float


@dataclass
class OperatorToken(Token):
    pass


@dataclass
class SpecialToken(Token):
    pass


@dataclass
class EOFToken(Token):
    pass


IDENTIFIER_CHARS = re.compile(r"[\w]|[\d]|_")
OPERATOR_CHARS = set("+-*/")
SPECIAL_CHARS = set("<>}{(),;=[]")


def tokenize(file: Path, program: str | None = None):
    tokens: List[Token] = []

    if program is None:
        with open(file, "r") as f:
            program = f.read()

    text = ""
    row = col = 1

    def flush():
        nonlocal col, row, text
        n = len(text)
        if n == 0:
            return

        true_col = col - n

        if text[0].isnumeric():
            value = float(text)
            tokens.append(NumberToken(file, row, true_col, text, value))
        else:
            tokens.append(IdentifierToken(file, row, true_col, text))

        text = ""

    for row, line in enumerate(program.splitlines()):
        # 1-indexed
        row += 1
        for col, char in enumerate(line):
            # 1-indexed
            col += 1
            if char == "#":
                # Comment
                break

            if IDENTIFIER_CHARS.match(char):
                text += char
                continue

            flush()

            if char == " ":
                continue

            if char in OPERATOR_CHARS:
                tokens.append(OperatorToken(file, row, col, char))
                continue
            elif char in SPECIAL_CHARS:
                tokens.append(SpecialToken(file, row, col, char))
                continue

            raise AssertionError(f"unhandled char {char} at ({row}, {col}) in \n{line}")

        col += 1
        flush()

    tokens.append(EOFToken(file, row, col, ""))

    return tokens
