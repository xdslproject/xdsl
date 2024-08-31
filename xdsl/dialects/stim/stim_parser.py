import re
from collections.abc import Callable, Sequence
from typing import TypeVar

from xdsl.dialects.builtin import ArrayAttr, IntAttr
from xdsl.dialects.stim import (
    QubitCoordsOp,
)
from xdsl.dialects.stim.ops import QubitAttr, QubitMappingAttr, StimCircuitOp
from xdsl.ir import Block, Operation, Region
from xdsl.utils.lexer import Input, Position
from xdsl.utils.str_enum import StrEnum

T = TypeVar("T")


class ParseError(Exception):
    position: Position
    message: str

    def __init__(self, position: Position, message: str) -> None:
        self.position = position
        self.message = message
        super().__init__(f"ParseError at {self.position}: {self.message}")


class InstructionSpelling(StrEnum):
    """
    Enum for the parse-able instructions in Stim.
    """

    COORD = "QUBIT_COORDS"


NEWLINE = re.compile(r"\n")
NOT_NEWLINE = re.compile(r"[^\n]*")
INDENT = re.compile(r"[ \t]*")
SPACE = re.compile(r"[ \t]")
ARG = re.compile(r"\d+(\.\d+)?")
# PAULI = re.compile("|".join(re.re.escape(p.value) for p in PauliSpelling))
TARGET = re.compile(r"\d+")
# GATE = re.compile("|".join(re.escape(p.value) for p in PrimitiveSpelling))
# INSTRUCTION = re.compile("|".join(re.escape(i.value) for i in InstructionSpelling))
NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")
"""
A regular expression for a name, - letter followed by a number of letters or numbers or underscores.
"""


class StimParser:
    input: Input
    pos: int

    def __init__(self, input: Input | str, pos: int = 0):
        if isinstance(input, str):
            input = Input(input, "<unknown>")
        self.input = input
        self.pos = pos

    @property
    def remaining(self) -> str:
        return self.input.content[self.pos :]

    # region Base parsing functions

    def parse_optional_chars(self, chars: str):
        if self.input.content.startswith(chars, self.pos):
            self.pos += len(chars)
            return chars

    def parse_optional_pattern(self, pattern: re.Pattern[str]):
        if (match := pattern.match(self.input.content, self.pos)) is not None:
            self.pos = match.regs[0][1]
            return self.input.content[match.pos : self.pos]

    # endregion

    # region Helpers
    def expect(self, message: str, parse: Callable[["StimParser"], T | None]) -> T:
        if (parsed := parse(self)) is None:
            raise ParseError(self.pos, message)
        return parsed

    def parse_one_of(
        self, StimParsers: Sequence[Callable[["StimParser"], T | None]]
    ) -> T | None:
        for StimParser in StimParsers:
            if (parsed := StimParser(self)) is not None:
                return parsed

    # endregion

    def parse_circuit(self) -> StimCircuitOp:
        """
        Greedily parse Stim dialect operations from a string formatted as a Stim file and return a StimCircuitOp
        containing the operations in its single block.
        """
        lines: list[Operation] = []
        while (op := self.parse_optional_operation()) is not None:
            lines.append(op)

        circuit_body = Region(Block(ops=lines))
        return StimCircuitOp(circuit_body, None)

    def parse_optional_comment(self) -> None:
        """
        Parse a comment if there, but this is not stored.
        """
        if self.parse_optional_chars("#") is None:
            return

        self.expect(
            "comment", lambda parser: parser.parse_optional_pattern(NOT_NEWLINE)
        )

    def _check_comment_and_newline(self) -> str | None:
        self.parse_optional_comment()
        return self.parse_optional_pattern(NEWLINE)

    def parse_optional_operation(
        self,
    ) -> Operation | None:
        """
        Stim lines have format:
            line ::= indent? (instruction | block_start | block_end)? (comment ::= `#' NotNewLine)? NEWLINE
        As this parser goes straight to operations, and stim comments, indentations and empty lines have no semantic meaning
        we look for operations, then skip straight through any lines without instructions or block starts or ends.
        """

        self.parse_optional_pattern(INDENT)
        op = self.parse_optional_instruction()  # TODO: add parsing for blocks

        # Skip comments and empty lines
        while self._check_comment_and_newline() is not None:
            pass
        return op

    # region Instruction parsing

    def parse_optional_instruction(self) -> Operation | None:
        """
        Parse instruction with format:
            instruction ::= name parens_arguments? targets
        """
        if (name := self.parse_optional_pattern(NAME)) is None:
            return None

        # raise ParseError(self.pos, "Expected name at the beginning of an instruction.")

        op = InstructionSpelling(name)
        parens = self.parse_optional_parens()
        if parens is None:
            parens = []
        targets = self.parse_targets()
        return self.build_operation(op, parens, targets)

    # region Parens parsing
    def parse_paren(self):
        self.parse_optional_pattern(INDENT)
        if (str_val := self.parse_optional_pattern(ARG)) is not None:
            arg = float(str_val)
            return arg

    def parse_optional_parens(self) -> list[float] | None:
        # Parse the opening bracket.
        if self.parse_optional_chars("(") is None:
            return None

        # Check that there is a first argument
        if (first := self.parse_paren()) is None:
            raise ParseError(self.pos, "Expected at least one argument in parentheses")
        args = [first]
        # Until the closing bracket is found:
        while self.parse_optional_chars(")") is None:
            self.parse_optional_pattern(INDENT)
            self.expect("comma", lambda parser: parser.parse_optional_chars(","))
            arg = self.expect("arg", lambda parser: parser.parse_paren())
            args.append(arg)

        return args

    # endregion

    # region Targets parsing

    def parse_target(self) -> QubitAttr | None:
        # Check for a space
        if self.parse_optional_pattern(INDENT) is None:
            raise ParseError(self.pos, "Targets must be separated by spacing.")
        if (str_val := self.parse_optional_pattern(TARGET)) is not None:
            target = int(str_val)
            return QubitAttr(target)

    def parse_targets(self) -> list[QubitAttr]:
        """
        Parse targets with form:
            targets = SPACE INDENTS target targets?
        TODO: the Stim documentation indicates that their parser requires at least one target per instruction - but their actual parser does not enforce this. Check incongruency.
        """
        # Check that there is a first target
        if (first := self.parse_target()) is None:
            raise ParseError(self.pos, "Expected at least one target")
        targets = [first]
        # Parse targets until no more are found
        while (target := self.parse_target()) is not None:
            targets.append(target)
        return targets

    # endregion

    def build_coord_parens(self, parens: list[float]) -> ArrayAttr[IntAttr]:
        coords = ArrayAttr([IntAttr(int(x)) for x in parens])
        return coords

    def build_operation(
        self, op: InstructionSpelling, parens: list[float], targets: Sequence[QubitAttr]
    ):
        match op:
            case InstructionSpelling.COORD:
                if len(targets) > 1:
                    raise ParseError(
                        self.pos, "QUBIT_COORDS can only act on one target"
                    )
                qubit = targets[0]
                coords = self.build_coord_parens(parens)
                mapping = QubitMappingAttr(coords, qubit)
                return QubitCoordsOp(mapping)

    # region Build operation

    # endregion

    # endregion
