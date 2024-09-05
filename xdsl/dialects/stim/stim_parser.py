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
from xdsl.dialects.stim.ops import (
    DepolarizingNoiseAttr,
    DetectorCoordsOp,
    MeasurementGateOp,
    PauliAttr,
    PauliOperatorEnum,
    ResetGateOp,
    SingleQubitCliffordsEnum,
    TickAnnotationOp,
    TwoQubitCliffordsEnum,
)
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
    TICK = "TICK"
    DETECTOR = "DETECTOR"


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


class TwoQubitUnitaryEnum(StrEnum):
    """
    Enum for the parse-able single qubit unitary gates in Stim.
    """

    SQRT_XX_DAG = "SQRT_XX_DAG"
    SQRT_YY_DAG = "SQRT_YY_DAG"
    SQRT_ZZ_DAG = "SQRT_ZZ_DAG"
    SQRT_XX = "SQRT_XX"
    SQRT_YY = "SQRT_YY"
    SQRT_ZZ = "SQRT_ZZ"

    CXSWAP = "CXSWAP"
    SWAPCX = "SWAPCX"
    CZSWAP = "CZSWAP"
    SWAPCZ = "SWAPCZ"
    iSWAP = "ISWAP"
    ISWAP_DAG = "ISWAP_DAG"

    SWAP = "SWAP"

    CNOT = "CNOT"
    CX = "CX"
    CY = "CY"
    CZ = "CZ"
    XCX = "XCX"
    XCY = "XCY"
    XCZ = "XCZ"
    YCX = "YCX"
    YCY = "YCY"
    YCZ = "YCZ"
    ZCX = "ZCX"
    ZCY = "ZCY"
    ZCZ = "ZCZ"


class MeasurementSpelling(StrEnum):
    MX = "MX"
    MY = "MY"
    MZ = "MZ"
    M = "M"


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
    "|".join(re.escape(gate.value) for gate in SingleQubitUnitaryEnum), re.IGNORECASE
)
TQGATE = re.compile(
    "|".join(re.escape(gate.value) for gate in TwoQubitUnitaryEnum), re.IGNORECASE
)
MEASUREMENT = re.compile(
    "|".join(re.escape(gate.value) for gate in MeasurementSpelling), re.IGNORECASE
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

    result_ssa_map: dict[int, SSAValue]
    seen_results: int

    def __init__(self, input: Input | str, pos: int = 0):
        if isinstance(input, str):
            input = Input(input, "<unknown>")
        self.input = input
        self.pos = pos
        self.seen_results = 0
        self.qubit_ssa_map = {}
        self.result_ssa_map = {}

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
    ) -> T | None:
        for StimParser in StimParsers:
            if (parsed := StimParser(self)) is not None:
                return parsed

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

    def parse_optional_instruction(self) -> tuple[list[Operation], Operation] | None:
        """
        Parse instruction with format:
            instruction ::= name (parens_arguments)? targets
        """

        # Check if any of the valid names parse
        op: tuple[list[Operation], Operation] | None = self.parse_one_of(
            [
                lambda parser: parser.parse_optional_annotation(),
                lambda parser: parser.parse_optional_measurement(),
                lambda parser: parser.parse_optional_gate(),
                lambda parser: parser.parse_optional_reset(),
                lambda parser: parser.parse_optional_noise(),
            ],
        )
        start_pos = self.pos
        # if any characters were not parsed that can be a name, the whole instruction is not known.
        if self.parse_optional_pattern(NAME) is not None:
            raise StimParseError(
                self.pos,
                f"`{self.input.content[start_pos:self.pos]}` is not a known instruction name.",
            )

        return op

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
            qref_val: SSAValue = new_qubit_op.results[0]
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
            return [], []
            # TODO: this potentially should have an error as part of the potential incongruency with stim's documentation - raise StimParseError(self.pos, "Expected at least one target")
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

    def parse_optional_noise(self):
        start_pos = self.pos
        qubit_type = 1
        if self.parse_optional_chars("DEPOLARIZE") is None:
            return None
        if self.parse_optional_chars("1") is None:
            if self.parse_optional_chars("2") is None:
                qubit_type = 2
                if self.parse_optional_pattern(NAME) is not None:
                    raise StimParseError(
                        self.pos,
                        f"Unknown operation {self.input.content[start_pos:self.pos]}",
                    )
        noise = 0
        if (parens := self.parse_optional_parens()) is not None:
            if len(parens) > 1:
                raise ValueError(
                    f"Gate DEPOLARIZE{qubit_type} was given {len(parens)} parens arguments {parens} but takes 0 or 1 parens arguments."
                )
            elif len(parens) == 1:
                noise = parens[0]
        extra_ops, targets = self.parse_targets()
        # TODO: Currently parsing is very lazy and just assigns any noise to an identity gate
        return extra_ops, CliffordGateOp(
            SingleQubitCliffordsEnum.Rotation,
            targets,
            noise=DepolarizingNoiseAttr(noise),
        )

    def parse_optional_measurement(self):
        start_pos = self.pos
        if self.parse_optional_chars("M") is None:
            return None
        if (gate_name := self.parse_optional_pattern(PAULI)) is None:
            if self.parse_optional_pattern(NAME) is not None:
                raise StimParseError(
                    self.pos,
                    f"Unknown operation {self.input.content[start_pos:self.pos]}",
                )
            else:
                gate_name = "Z"
        noise = 0
        if (parens := self.parse_optional_parens()) is not None:
            if len(parens) > 1:
                raise ValueError(
                    f"Gate {gate_name} was given {len(parens)} parens arguments {parens} but takes 0 or 1 parens arguments."
                )
            elif len(parens) == 1:
                noise = parens[0]
        extra_ops, targets = self.parse_targets()
        measurement_op = MeasurementGateOp.create(
            operands=targets,
            result_types=[qref.QubitAttr()] * len(targets),
            properties={
                "pauli_modifier": PauliAttr(PauliOperatorEnum(gate_name)),
                "noise": DepolarizingNoiseAttr(noise),
            },
        )
        for result in measurement_op.results:
            self.result_ssa_map[self.seen_results] = result
            self.seen_results += 1
        return extra_ops, measurement_op

    def parse_optional_reset(self):
        start_pos = self.pos
        if self.parse_optional_chars("R") is None:
            return None
        if (gate_name := self.parse_optional_pattern(PAULI)) is None:
            if self.parse_optional_pattern(NAME) is not None:
                raise StimParseError(
                    self.pos,
                    f"Unknown operation {self.input.content[start_pos:self.pos]}",
                )
            else:
                gate_name = "Z"
        if ((parens := self.parse_optional_parens()) is not None) and len(parens) > 1:
            raise ValueError(
                f"Gate {gate_name} was given {len(parens)} parens arguments {parens} but takes 0 or 1 parens arguments."
            )
        extra_ops, targets = self.parse_targets()
        reset_op = ResetGateOp.create(
            operands=targets,
            result_types=[qref.QubitAttr()] * len(targets),
            properties={"pauli_modifier": PauliAttr(PauliOperatorEnum(gate_name))},
        )
        return extra_ops, reset_op

    def parse_optional_gate_name(
        self,
    ) -> (
        tuple[
            (SingleQubitCliffordsEnum | TwoQubitCliffordsEnum),
            (PauliOperatorEnum | None),
            (PauliOperatorEnum | None),
            bool,
            bool,
        ]
        | None
    ):
        # TODO: Finish this version of the parser to remove ugly enum matching. In particular, make the errors return a complaint that parses to the next whitespace.
        if self.parse_optional_chars("SQRT_") is not None:
            pauli = self.parse_optional_pattern(PAULI)
            if pauli is None:
                raise StimParseError(self.pos, "Expected pauli after SQRT_")
            pauli2 = self.parse_optional_pattern(PAULI)
            dag = False
            if self.parse_optional_chars("_DAG") is not None:
                dag = True
            if pauli2 is None:
                return (
                    SingleQubitCliffordsEnum.Rotation,
                    PauliOperatorEnum(pauli),
                    None,
                    True,
                    dag,
                )
            else:
                if pauli2 != pauli:
                    raise StimParseError(
                        self.pos, f"SQRT_{pauli}{pauli2} is not a known operation"
                    )
                return (
                    TwoQubitCliffordsEnum.Both_Pauli,
                    PauliOperatorEnum(pauli),
                    PauliOperatorEnum(pauli2),
                    True,
                    dag,
                )
        elif self.parse_optional_chars("C"):
            pauli = self.parse_optional_pattern(PAULI)
            if pauli is None:
                if self.parse_optional_chars("NOT") is None:
                    raise StimParseError(self.pos, "Expected pauli after C")
                return (
                    TwoQubitCliffordsEnum.Ctrl,
                    PauliOperatorEnum.Z,
                    PauliOperatorEnum.X,
                    False,
                    False,
                )
            if self.parse_optional_chars("SWAP"):
                return (
                    TwoQubitCliffordsEnum.Midswap,
                    PauliOperatorEnum(pauli),
                    None,
                    False,
                    False,
                )
            return (
                TwoQubitCliffordsEnum.Ctrl,
                PauliOperatorEnum.Z,
                PauliOperatorEnum(pauli),
                False,
                False,
            )
        elif self.parse_optional_chars("H"):
            if self.parse_optional_chars("_XY"):
                return (
                    SingleQubitCliffordsEnum.BiAxisRotation,
                    PauliOperatorEnum.Z,
                    None,
                    False,
                    False,
                )
            if self.parse_optional_chars("_XZ"):
                return (
                    SingleQubitCliffordsEnum.BiAxisRotation,
                    PauliOperatorEnum.Y,
                    None,
                    False,
                    False,
                )
            if self.parse_optional_chars("_YZ"):
                return (
                    SingleQubitCliffordsEnum.BiAxisRotation,
                    PauliOperatorEnum.X,
                    None,
                    False,
                    False,
                )
            return (
                SingleQubitCliffordsEnum.BiAxisRotation,
                PauliOperatorEnum.Y,
                None,
                False,
                False,
            )
        elif (pauli := self.parse_optional_pattern(PAULI)) is not None:
            if self.parse_optional_chars("C"):
                pauli = self.parse_optional_pattern(PAULI)
                if pauli is None:
                    if self.parse_optional_chars("NOT") is None:
                        raise StimParseError(self.pos, "Expected pauli after C")
                    return (
                        TwoQubitCliffordsEnum.Ctrl,
                        PauliOperatorEnum.Z,
                        PauliOperatorEnum.X,
                        False,
                        False,
                    )
                return (
                    TwoQubitCliffordsEnum.Ctrl,
                    PauliOperatorEnum.Z,
                    PauliOperatorEnum(pauli),
                    False,
                    False,
                )
            return (
                SingleQubitCliffordsEnum.Rotation,
                PauliOperatorEnum(pauli),
                None,
                False,
                False,
            )
        i = False
        if self.parse_optional_chars("I") is not None:
            i = True
        if self.parse_optional_chars("SWAP") is not None:
            if i:
                dag = False
                if self.parse_optional_chars("_DAG") is not None:
                    dag = True
                return (
                    TwoQubitCliffordsEnum.Midswap,
                    PauliOperatorEnum.Y,
                    None,
                    False,
                    dag,
                )
            if self.parse_optional_chars("CX") is not None:
                return (
                    TwoQubitCliffordsEnum.Midswap,
                    PauliOperatorEnum.X,
                    None,
                    False,
                    False,
                )
            elif self.parse_optional_chars("CZ") is not None:
                return (
                    TwoQubitCliffordsEnum.Midswap,
                    PauliOperatorEnum.Z,
                    None,
                    False,
                    False,
                )
            return TwoQubitCliffordsEnum.Swap, None, None, False, False
        if i:
            return SingleQubitCliffordsEnum.Rotation, None, None, False, False
        if self.parse_optional_chars("S") is not None:
            dag = False
            if self.parse_optional_chars("_DAG") is not None:
                dag = True
            return (
                SingleQubitCliffordsEnum.Rotation,
                PauliOperatorEnum.Z,
                None,
                True,
                dag,
            )
        return None

    def parse_optional_gate(self):
        name_start_pos = self.pos
        if (gate_options := self.parse_optional_gate_name()) is None:
            return None

        # if any characters were not parsed that can be a name, the whole instruction is not known.
        if self.parse_optional_pattern(NAME) is not None:
            raise StimParseError(
                self.pos,
                f"`{self.input.content[name_start_pos:self.pos]}` is not a known instruction name.",
            )

        name_end_pos = self.pos
        parens = self.parse_optional_parens()
        if parens is not None:
            raise StimParseError(
                name_end_pos,
                f"Gate {self.input.content[name_start_pos: name_end_pos]} was given parens arguments ({len(parens)}) but expects (0) arguments.",
            )
        extra_ops, targets = self.parse_targets()
        if isinstance(gate_options[0], TwoQubitCliffordsEnum):
            if len(targets) % 2 != 0:
                raise StimParseError(
                    self.pos,
                    f"Gate {self.input.content[name_start_pos: name_end_pos]} was given an odd number of targets but expects.",
                )
            for ctrl, target in zip(targets[0::2], targets[1::2]):
                if ctrl == target:
                    raise StimParseError(
                        self.pos,
                        f"The two qubit gate {self.input.content[name_start_pos: name_end_pos]} was applied to a target pair with the same target ({ctrl}) twice. Gates can't interact targets with themselves.",
                    )

        return extra_ops, CliffordGateOp(
            gate_options[0],
            targets,
            gate_options[1],
            gate_options[2],
            gate_options[4],
            gate_options[3],
        )

    def parse_optional_rec(self) -> SSAValue | None:
        """
        Parse an argument passed to an instruction in parentheses with format:
            rec ::= rec[-`int`]
        """
        # Check for a space
        space = self.parse_optional_pattern(SPACE)
        if self.parse_optional_chars("rec[-") is None:
            return None
        if space is None:
            raise StimParseError(self.pos, "Targets must be separated by spacing.")
        if (str_val := self.parse_optional_pattern(TARGET)) is not None:
            arg = int(str_val)
            if arg > self.seen_results:
                raise StimParseError(self.pos, "Record argument out of range.")
            if self.parse_optional_chars("]") is not None:
                return self.result_ssa_map[self.seen_results - arg]
        raise StimParseError(
            self.pos,
            "Target started with 'r' but wasn't a record argument like 'rec[-1]'",
        )

    def parse_optional_recs(self) -> list[SSAValue] | None:
        """
        Parse an optional parenthesis or recs with optional arguments with format:
            parens ::= `(` (INDENT* rec INDENT* (',' INDENT* rec INDENT*)*)? `)`

        TODO: The Stim documentation uses this format:
            <PARENS_ARGUMENTS> ::= '(' <ARGUMENTS> ')'
            <ARGUMENTS> ::= /[ \t]*/ <ARG> /[ \t]*/ (',' <ARGUMENTS>)?

            but its implementation accepts empty arguments - this is kept here to be consistent.

        """
        if (rec := self.parse_optional_rec()) is None:
            return None
        recs: list[SSAValue] = [rec]
        # whilst recs are found:
        # Parse targets until no more are found
        while (rec := self.parse_optional_rec()) is not None:
            recs.append(rec)
        return recs

    def parse_optional_annotation(self) -> tuple[list[Operation], Operation] | None:
        name_start_pos = self.pos
        if (annotation := self.parse_optional_pattern(ANNOTATION)) is None:
            return None
        # if any characters were not parsed that can be a name, the whole instruction is not known.
        if self.parse_optional_pattern(NAME) is not None:
            raise StimParseError(
                self.pos,
                f"`{self.input.content[name_start_pos:self.pos]}` is not a known instruction name.",
            )

        parens = self.parse_optional_parens()
        if parens == []:
            # In line with the behaviour of the Stim parser, parsing an empty parens argument gives a paren with 0 in.
            parens = [0.0]
        elif parens is None:
            # In line with the behaviour of the Stim parser, parsing no parens argument gives an empty list.
            parens = []

        if AnnotationEnum(annotation) == AnnotationEnum.DETECTOR:
            coords = self.build_parens(parens)
            if (recs := self.parse_optional_recs()) is None:
                return [], DetectorCoordsOp([], QubitMappingAttr(coords))
            return [], DetectorCoordsOp(recs, QubitMappingAttr(coords))

        extra_ops, targets = self.parse_targets()
        match AnnotationEnum(annotation):
            case AnnotationEnum.COORD:
                qubit = targets[0]
                coords = self.build_parens(parens)
                mapping = QubitMappingAttr(coords)
                return extra_ops, QubitCoordsOp(qubit, mapping)
            case AnnotationEnum.TICK:
                if parens != []:
                    raise StimParseError(
                        self.pos,
                        f"TICK operation expects no parens arguments but was given {parens}.",
                    )
                if targets != []:
                    raise StimParseError(
                        self.pos,
                        f"TICK operation expects no targets but was given targets {len(targets)}.",
                    )
                return [], TickAnnotationOp()
            case _:
                raise StimParseError(
                    self.pos, f"Annotation {annotation} not currently supported."
                )

    # endregion

    # endregion
