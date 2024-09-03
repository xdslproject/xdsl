import re
from collections.abc import Callable, Sequence
from typing import TypeVar

from xdsl.dialects import qref
from xdsl.dialects.builtin import ArrayAttr, FloatData, IntAttr
from xdsl.dialects.stim import (
    CliffordGateOp,
    QubitCoordsOp,
    QubitMappingAttr,
    StimCircuitOp,
)
from xdsl.dialects.stim.ops import PauliOperatorEnum, SingleQubitCliffordsEnum
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.utils.lexer import Input, Position
from xdsl.utils.str_enum import StrEnum

T = TypeVar("T")
T1 = TypeVar("T1")


class StimParseError(Exception):
    position: Position
    message: str

    def __init__(self, position: Position, message: str) -> None:
        self.position = position
        self.message = message
        super().__init__(f"StimParseError at {self.position}: {self.message}")


class AnnotationEnum(StrEnum):
    """
    Enum for the parse-able instructions in Stim.
    """

    COORD = "QUBIT_COORDS"


class SingleQubitUnitaryEnum(StrEnum):
    """
    Enum for the parse-able single qubit unitary gates in Stim.
    """

    IDENTITY = "I"
    HXY = "H_XY"
    HYZ = "H_YZ"
    HXZ = "H_XZ"
    H = "H"
    SQRT_X_DAG = "SQRT_X_DAG"
    SQRT_Y_DAG = "SQRT_Y_DAG"
    SQRT_Z_DAG = "SQRT_Z_DAG"
    SQRT_X = "SQRT_X"
    SQRT_Y = "SQRT_Y"
    SQRT_Z = "SQRT_Z"
    S_DAG = "S_DAG"
    S = "S"
    C_XYZ = "C_XYZ"
    C_ZYX = "C_ZYX"


NEWLINE = re.compile(r"\n")
NOT_NEWLINE = re.compile(r"[^\n]*")
INDENT = re.compile(r"[ \t]*")
SPACE = re.compile(r"[ \t]")
ARG = re.compile(r"\d+(\.\d+)?")
PAULI = re.compile(
    "|".join(re.escape(p.value) for p in PauliOperatorEnum), re.IGNORECASE
)
TARGET = re.compile(r"\d+")
SQGATE = re.compile(
    "|".join(re.escape(p.value) for p in SingleQubitUnitaryEnum), re.IGNORECASE
)
ANNOTATION = re.compile(
    "|".join(re.escape(i.value) for i in AnnotationEnum), re.IGNORECASE
)
NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")
"""
A regular expression for a name: a letter followed by a number of letters or numbers or underscores.
"""


class StimParser:
    input: Input
    pos: int

    qubit_ssa_map: dict[int, SSAValue]

    def __init__(self, input: Input | str, pos: int = 0):
        if isinstance(input, str):
            input = Input(input, "<unknown>")
        self.input = input
        self.pos = pos
        self.qubit_ssa_map = {}

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
            raise StimParseError(self.pos, message)
        return parsed

    def parse_one_of(
        self,
        StimParsers: Sequence[Callable[["StimParser"], T | None]],
        builders: Sequence[Callable[[T], T1]],
    ) -> T1 | None:
        for StimParser, build in zip(StimParsers, builders):
            if (parsed := StimParser(self)) is not None:
                return build(parsed)

    # endregion

    def parse_circuit(self) -> StimCircuitOp:
        """
        Parse Stim dialect operations from a string formatted as a Stim file and return a StimCircuitOp
        containing the operations in its single block.

        Circuits have format:
            circuit         ::= (line)*

        Collect by operations instead of by line to skip any lines that are not converted into operations.
        """
        lines: list[Operation] = []
        while (op := self.parse_optional_operation()) is not None:
            lines += op[0]
            lines.append(op[1])

        circuit_body = Region(Block(ops=lines))
        return StimCircuitOp(circuit_body)

    def parse_optional_comment(self) -> None:
        """
        Parse a comment if there, but this is not stored.

        Comments have format:
            comment ::= `#` NOT_NEWLINE
        """
        if self.parse_optional_chars("#") is None:
            return

        self.expect(
            "comment", lambda parser: parser.parse_optional_pattern(NOT_NEWLINE)
        )

    def _check_comment_and_newline(self) -> str | None:
        """
        Consume an optional comment and newline, returning if a newline has occured.
        """
        self.parse_optional_comment()
        return self.parse_optional_pattern(NEWLINE)

    def parse_optional_operation(
        self,
    ) -> tuple[list[Operation], Operation] | None:
        """
        Stim lines have format:
            line ::= indent? (instruction | block_start | block_end)? (comment ::= `#' NotNewLine)? NEWLINE
        As this parser goes straight to operations, and stim comments, indentations and empty lines have no semantic meaning
        we look for operations, then skip straight through any lines without instructions or block starts or ends.

        Operations are given by instructions, or the a block containing instructions.

        As stim operations are able to introduce new qubits non-explicitly, parsing an operation may return additional
        allocation operations to explicitly allocate the ssa-values for the qubits.
        """

        self.parse_optional_pattern(INDENT)
        ops = self.parse_optional_instruction()  # TODO: add parsing for blocks

        # Skip comments and empty lines
        while self._check_comment_and_newline() is not None:
            pass
        return ops

    # region Instruction parsing
    def parse_optional_name(self):
        if (
            op := self.parse_one_of(
                [
                    lambda parser: parser.parse_optional_pattern(PAULI),
                    lambda parser: parser.parse_optional_pattern(SQGATE),
                    lambda parser: parser.parse_optional_pattern(ANNOTATION),
                ],
                [
                    lambda str: PauliOperatorEnum(str),
                    lambda str: SingleQubitUnitaryEnum(str),
                    lambda str: AnnotationEnum(str),
                ],
            )
        ) is None:
            if n := self.parse_optional_pattern(NAME) is not None:
                raise StimParseError(
                    self.pos, f"`{n}` is not a known instruction name."
                )
            return None
        return op

    def parse_optional_instruction(self) -> tuple[list[Operation], Operation] | None:
        """
        Parse instruction with format:
            instruction ::= name (parens_arguments)? targets
        """

        # Check if any of the valid names pass
        if (op := self.parse_optional_name()) is None:
            return None
        parens = self.parse_optional_parens()
        if parens is None:
            parens = []
        extra_ops, targets = self.parse_targets()
        return (extra_ops, self.build_operation(op, parens, targets))

    # region Parens parsing

    def parse_optional_paren(self):
        """
        Parse an argument passed to an instruction in parentheses with format:
            arg ::= double
        """
        self.parse_optional_pattern(INDENT)
        if (str_val := self.parse_optional_pattern(ARG)) is not None:
            arg = float(str_val)
            return arg

    def parse_optional_parens(self) -> list[float] | None:
        """
        Parse an optional parenthesis with optional arguments with format:
            parens ::= `(` (INDENT* arg INDENT* (',' INDENT* arg INDENT*)*)? `)`

        TODO: The Stim documentation uses this format:
            <PARENS_ARGUMENTS> ::= '(' <ARGUMENTS> ')'
            <ARGUMENTS> ::= /[ \t]*/ <ARG> /[ \t]*/ (',' <ARGUMENTS>)?

            but its implementation accepts empty arguments - this is kept here to be consistent.

        """
        # Parse the opening bracket.
        if self.parse_optional_chars("(") is None:
            return None
        if self.parse_optional_chars(")") is not None:
            return []
        # Check if an argument exists.
        args: list[float] = [
            self.expect("arg", lambda parser: parser.parse_optional_paren())
        ]
        # Until the closing bracket is found:
        while self.parse_optional_chars(")") is None:
            self.parse_optional_pattern(INDENT)
            self.expect("comma", lambda parser: parser.parse_optional_chars(","))
            arg = self.expect("arg", lambda parser: parser.parse_optional_paren())
            args.append(arg)

        return args

    # endregion

    # region Targets parsing

    def parse_optional_target(self) -> tuple[Operation, SSAValue] | SSAValue | None:
        """
        Parse a target with format:
            target ::= <QUBIT_TARGET> | <MEASUREMENT_RECORD_TARGET> | <SWEEP_BIT_TARGET> | <PAULI_TARGET> | <COMBINER_TARGET>
        TODO: Currently only supports target ::= qubit_target as the other relevant operations are not yet supported

        If a target has not been seen before, then it must be provided a new SSA-value that is the qubit reference. This is currently
        implemented by allocating a new qubit.
        TODO: this ought to be revisited when the lattice surgery dialect is written.
        """
        # Check for a space
        space = self.parse_optional_pattern(SPACE)
        self.parse_optional_pattern(INDENT)
        if (str_val := self.parse_optional_pattern(TARGET)) is not None:
            if space is None:
                raise StimParseError(self.pos, "Targets must be separated by spacing.")
            target = int(str_val)
            # If the target has already been allocated, return the ssa value associated with it.
            if target in self.qubit_ssa_map:
                return self.qubit_ssa_map[target]
            new_qubit_op = qref.QRefAllocOp(1)
            qref_val = new_qubit_op.results[0]
            self.qubit_ssa_map[target] = qref_val
            return (new_qubit_op, qref_val)

    def parse_targets(self) -> tuple[list[Operation], list[SSAValue]]:
        """
        Parse targets with format:
            targets = SPACE INDENTS target targets?
        TODO: the Stim documentation indicates that their parser requires at least one target per instruction - but their actual parser does not enforce this. Check incongruency.
        """
        # Check that there is a first target
        if (first := self.parse_optional_target()) is None:
            raise StimParseError(self.pos, "Expected at least one target")
        targets: list[SSAValue] = []
        extra_alloc_ops: list[Operation] = []
        if isinstance(first, tuple):
            extra_alloc_ops.append(first[0])
            targets.append(first[1])
        else:
            targets.append(first)
        # Parse targets until no more are found
        while (target := self.parse_optional_target()) is not None:
            if isinstance(target, tuple):
                extra_alloc_ops.append(target[0])
                targets.append(target[1])
            else:
                targets.append(target)
        return (extra_alloc_ops, targets)

    # endregion

    # region Build operation

    def build_parens(self, parens: list[float]) -> ArrayAttr[FloatData | IntAttr]:
        """
        Convert a list of parens into an ArrayAttr.
        """
        args = [
            (IntAttr(int(arg))) if (arg.is_integer()) else (FloatData(arg))
            for arg in parens
        ]
        coords = ArrayAttr(args)
        return coords

    def build_operation(
        self,
        op: AnnotationEnum | PauliOperatorEnum | SingleQubitUnitaryEnum,
        parens: list[float],
        targets: Sequence[SSAValue],
    ):
        """
        Build the operation corresponding to the name, parens, and targets found by the parser.
        """

        if isinstance(op, AnnotationEnum):
            match op:
                case AnnotationEnum.COORD:
                    if parens == []:
                        # In line with the behaviour of the Stim parser, parsing an empty parens argument gives a paren with 0 in.
                        parens = [0.0]
                    qubit = targets[0]
                    coords = self.build_parens(parens)
                    mapping = QubitMappingAttr(coords)
                    return QubitCoordsOp(qubit, mapping)
        if isinstance(op, PauliOperatorEnum):
            if parens != []:
                raise StimParseError(
                    self.pos,
                    f"Gate {op} was given parens arguments ({parens.count}) but expects (0) arguments.",
                )
            return CliffordGateOp(SingleQubitCliffordsEnum.Rotation, targets, op)
        if isinstance(op, SingleQubitUnitaryEnum):
            if parens != []:
                raise StimParseError(
                    self.pos,
                    f"Gate {op} was given parens arguments ({parens.count}) but expects (0) arguments.",
                )
            match op:
                case SingleQubitUnitaryEnum.IDENTITY:
                    return CliffordGateOp(SingleQubitCliffordsEnum.Rotation, targets)
                case SingleQubitUnitaryEnum.H | SingleQubitUnitaryEnum.HXZ:
                    return CliffordGateOp(
                        SingleQubitCliffordsEnum.BiAxisRotation, targets
                    )
                case SingleQubitUnitaryEnum.HXY:
                    return CliffordGateOp(
                        SingleQubitCliffordsEnum.BiAxisRotation,
                        targets,
                        PauliOperatorEnum.Z,
                    )
                case SingleQubitUnitaryEnum.HYZ:
                    return CliffordGateOp(
                        SingleQubitCliffordsEnum.BiAxisRotation,
                        targets,
                        PauliOperatorEnum.X,
                    )
                case SingleQubitUnitaryEnum.S | SingleQubitUnitaryEnum.SQRT_Z:
                    return CliffordGateOp(
                        SingleQubitCliffordsEnum.Rotation,
                        targets,
                        PauliOperatorEnum.Z,
                        is_sqrt=True,
                    )
                case SingleQubitUnitaryEnum.S_DAG | SingleQubitUnitaryEnum.SQRT_Z_DAG:
                    return CliffordGateOp(
                        SingleQubitCliffordsEnum.Rotation,
                        targets,
                        PauliOperatorEnum.Z,
                        is_dag=True,
                        is_sqrt=True,
                    )
                case SingleQubitUnitaryEnum.SQRT_X:
                    return CliffordGateOp(
                        SingleQubitCliffordsEnum.Rotation,
                        targets,
                        PauliOperatorEnum.X,
                        is_sqrt=True,
                    )
                case SingleQubitUnitaryEnum.SQRT_X_DAG:
                    return CliffordGateOp(
                        SingleQubitCliffordsEnum.Rotation,
                        targets,
                        PauliOperatorEnum.X,
                        is_dag=True,
                        is_sqrt=True,
                    )
                case SingleQubitUnitaryEnum.SQRT_Y:
                    return CliffordGateOp(
                        SingleQubitCliffordsEnum.Rotation,
                        targets,
                        PauliOperatorEnum.Y,
                        is_sqrt=True,
                    )
                case SingleQubitUnitaryEnum.SQRT_Y_DAG:
                    return CliffordGateOp(
                        SingleQubitCliffordsEnum.Rotation,
                        targets,
                        PauliOperatorEnum.Y,
                        is_dag=True,
                        is_sqrt=True,
                    )
                case _:
                    raise StimParseError(self.pos, f"Parsing not implemented for {op}")

    # endregion

    # endregion
