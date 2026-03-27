from abc import ABC
from collections.abc import Sequence
from io import StringIO
from typing import ClassVar

from xdsl.dialects.builtin import ArrayAttr, FloatData, IntAttr, f64
from xdsl.dialects.stim.stim_printer_parser import StimPrintable, StimPrinter
from xdsl.ir import ParametrizedAttribute, Region, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    region_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class QubitAttr(StimPrintable, ParametrizedAttribute, TypeAttribute):
    """
    Type for a single qubit.
    """

    name = "stim.qubit"

    qubit: IntAttr

    def __init__(self, qubit: int | IntAttr) -> None:
        if not isinstance(qubit, IntAttr):
            qubit = IntAttr(qubit)
        super().__init__(qubit)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[IntAttr]:
        with parser.in_angle_brackets():
            qubit = parser.parse_integer(allow_negative=False, allow_boolean=False)
            return (IntAttr(qubit),)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_int(self.qubit.data)

    def print_stim(self, printer: StimPrinter):
        printer.print_string(f"{self.qubit.data}")


@irdl_attr_definition
class QubitMappingAttr(StimPrintable, ParametrizedAttribute):
    """
    This attribute provides a way to indicate the required connectivity or layout of
    `physical` qubits.

    It consists of two parameters:
     1. A co-ordinate array (currently it only anticipates a pair of qubits, but this is
     not fixed)
     2. A value associated with a qubit referred to in a circuit.

    The co-ordinates may be used as a physical address of a qubit, or the relative
    address with respect to some known physical address.

    Operations that attach this as a property may represent the lattice-like structure
    of a physical quantum computer by having a property with an
    ArrayAttr[QubitCoordsAttr].
    """

    name = "stim.qubit_coord"

    coords: ArrayAttr[FloatData | IntAttr]
    qubit_name: QubitAttr

    def __init__(
        self,
        coords: list[float] | ArrayAttr[FloatData | IntAttr],
        qubit_name: int | QubitAttr,
    ) -> None:
        if not isinstance(qubit_name, QubitAttr):
            qubit_name = QubitAttr(qubit_name)
        if not isinstance(coords, ArrayAttr):
            coords = ArrayAttr(
                (IntAttr(int(arg))) if (type(arg) is int) else (FloatData(arg))
                for arg in coords
            )
        super().__init__(coords, qubit_name)

    @classmethod
    def parse_parameters(
        cls, parser: AttrParser
    ) -> tuple[ArrayAttr[FloatData | IntAttr], QubitAttr]:
        parser.parse_punctuation("<")
        coords = parser.parse_comma_separated_list(
            delimiter=parser.Delimiter.PAREN,
            parse=lambda: (
                IntAttr(x)
                if type(x := parser.parse_number(allow_boolean=False)) is int
                else FloatData(x)
            ),
        )
        parser.parse_punctuation(",")
        qubit = parser.parse_attribute()
        if not isinstance(qubit, QubitAttr):
            parser.raise_error("Expected qubit attr", at_position=parser.pos)
        parser.parse_punctuation(">")
        return (ArrayAttr(coords), qubit)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            with printer.in_parens():
                printer.print_list(
                    self.coords,
                    lambda c: (
                        printer.print_int(c.data)
                        if isinstance(c, IntAttr)
                        else printer.print_float(c.data, f64)
                    ),
                )
            printer.print_string(", ")
            printer.print_attribute(self.qubit_name)

    def print_stim(self, printer: StimPrinter):
        printer.print_attribute(self.coords)
        self.qubit_name.print_stim(printer)


@irdl_op_definition
class StimCircuitOp(StimPrintable, IRDLOperation):
    """
    Base operation containing a stim program
    """

    name = "stim.circuit"

    body = region_def("single_block")

    qubitlayout = opt_prop_def(ArrayAttr[QubitMappingAttr])

    assembly_format = "(`qubitlayout` $qubitlayout^)? attr-dict-with-keyword $body"

    def __init__(self, body: Region, qubitlayout: None | ArrayAttr[QubitMappingAttr]):
        super().__init__(regions=[body], properties={"qubitlayout": qubitlayout})

    def verify(self, verify_nested_ops: bool = True) -> None:
        return

    def print_stim(self, printer: StimPrinter):
        for op in self.body.block.ops:
            printer.print_op(op)
            printer.print_string("\n")
        printer.print_string("")

    def stim(self) -> str:
        io = StringIO()
        printer = StimPrinter(io)
        self.print_stim(printer)
        res = io.getvalue()
        return res


"""
Annotation Operations

Stim contains a number of `Annotations` - instructions which do not affect the operational semantics of the stim circuit -
but may provide useful information about the circuit being run or how decoding should be done.

These are essentially code-directives for compiler analyses on the circuit.

Here each is attached as an attribute instead - but as they may appear in the code, are also given operations that can
drive the change of a value and be used to direct printing of stim circuits.
"""


class AnnotationOp(StimPrintable, IRDLOperation, ABC):
    """
    Base Annotation operation.

    This is used to indicate operations that are stim annotations,
    these do not have operational semantics,
    so this will be used during transforms to ignore these operations.
    """


@irdl_op_definition
class QubitCoordsOp(AnnotationOp):
    """
    Annotation operation that assigns a qubit reference to a coordinate.
    """

    name = "stim.assign_qubit_coord"

    qubitmapping = prop_def(QubitMappingAttr)

    assembly_format = "$qubitmapping attr-dict"

    def __init__(self, qubitmapping: QubitMappingAttr):
        super().__init__(properties={"qubitmapping": qubitmapping})

    def print_stim(self, printer: StimPrinter) -> None:
        printer.print_string("QUBIT_COORDS")
        self.qubitmapping.print_stim(printer)


@irdl_op_definition
class TickOp(AnnotationOp):
    """
    Annotation operation that marks a time step boundary.
    """

    name = "stim.tick"

    assembly_format = "attr-dict"

    def __init__(self):
        super().__init__()

    def print_stim(self, printer: StimPrinter) -> None:
        printer.print_string("TICK")


"""
Gate Operations

Stim contains a number of gate operations that apply quantum gates to qubits.
"""


class SingleQubitGateOp(StimPrintable, IRDLOperation, ABC):
    """
    Base class for single-qubit gate operations.

    These operations apply a single-qubit gate to one or more qubit targets.
    """

    STIM_NAME: ClassVar[str]

    targets = prop_def(ArrayAttr[QubitAttr])

    assembly_format = "$targets attr-dict"

    def __init__(self, targets: ArrayAttr[QubitAttr] | Sequence[QubitAttr | int]):
        if not isinstance(targets, ArrayAttr):
            targets = ArrayAttr(
                QubitAttr(t) if isinstance(t, int) else t for t in targets
            )
        super().__init__(properties={"targets": targets})

    def print_stim(self, printer: StimPrinter) -> None:
        printer.print_string(self.STIM_NAME)
        for target in self.targets:
            printer.print_string(" ")
            target.print_stim(printer)


@irdl_op_definition
class HOp(SingleQubitGateOp):
    name = "stim.h"
    STIM_NAME: ClassVar[str] = "H"


@irdl_op_definition
class SOp(SingleQubitGateOp):
    name = "stim.s"
    STIM_NAME: ClassVar[str] = "S"


@irdl_op_definition
class SDagOp(SingleQubitGateOp):
    name = "stim.s_dag"
    STIM_NAME: ClassVar[str] = "S_DAG"


@irdl_op_definition
class XOp(SingleQubitGateOp):
    name = "stim.x"
    STIM_NAME: ClassVar[str] = "X"


@irdl_op_definition
class YOp(SingleQubitGateOp):
    name = "stim.y"
    STIM_NAME: ClassVar[str] = "Y"


@irdl_op_definition
class ZOp(SingleQubitGateOp):
    name = "stim.z"
    STIM_NAME: ClassVar[str] = "Z"


@irdl_op_definition
class IOp(SingleQubitGateOp):
    name = "stim.i"
    STIM_NAME: ClassVar[str] = "I"


@irdl_op_definition
class SqrtXOp(SingleQubitGateOp):
    name = "stim.sqrt_x"
    STIM_NAME: ClassVar[str] = "SQRT_X"


@irdl_op_definition
class SqrtXDagOp(SingleQubitGateOp):
    name = "stim.sqrt_x_dag"
    STIM_NAME: ClassVar[str] = "SQRT_X_DAG"


@irdl_op_definition
class SqrtYOp(SingleQubitGateOp):
    name = "stim.sqrt_y"
    STIM_NAME: ClassVar[str] = "SQRT_Y"


@irdl_op_definition
class SqrtYDagOp(SingleQubitGateOp):
    name = "stim.sqrt_y_dag"
    STIM_NAME: ClassVar[str] = "SQRT_Y_DAG"


class TwoQubitGateOp(StimPrintable, IRDLOperation, ABC):
    """
    Base class for two-qubit gate operations.

    These operations apply a two-qubit gate to pairs of qubit targets.
    The number of targets must be even.
    """

    STIM_NAME: ClassVar[str]

    targets = prop_def(ArrayAttr[QubitAttr])

    assembly_format = "$targets attr-dict"

    def __init__(self, targets: ArrayAttr[QubitAttr] | Sequence[QubitAttr | int]):
        if not isinstance(targets, ArrayAttr):
            targets = ArrayAttr(
                QubitAttr(t) if isinstance(t, int) else t for t in targets
            )
        super().__init__(properties={"targets": targets})

    def verify_(self) -> None:
        if len(self.targets) % 2 != 0:
            raise VerifyException(
                f"{self.STIM_NAME} requires an even number of targets, "
                f"got {len(self.targets)}"
            )

    def print_stim(self, printer: StimPrinter) -> None:
        printer.print_string(self.STIM_NAME)
        for target in self.targets:
            printer.print_string(" ")
            target.print_stim(printer)


@irdl_op_definition
class CXOp(TwoQubitGateOp):
    name = "stim.cx"
    STIM_NAME: ClassVar[str] = "CX"


@irdl_op_definition
class CYOp(TwoQubitGateOp):
    name = "stim.cy"
    STIM_NAME: ClassVar[str] = "CY"


@irdl_op_definition
class CZOp(TwoQubitGateOp):
    name = "stim.cz"
    STIM_NAME: ClassVar[str] = "CZ"


@irdl_op_definition
class SwapOp(TwoQubitGateOp):
    name = "stim.swap"
    STIM_NAME: ClassVar[str] = "SWAP"


@irdl_op_definition
class ISwapOp(TwoQubitGateOp):
    name = "stim.iswap"
    STIM_NAME: ClassVar[str] = "ISWAP"


@irdl_op_definition
class ISwapDagOp(TwoQubitGateOp):
    name = "stim.iswap_dag"
    STIM_NAME: ClassVar[str] = "ISWAP_DAG"


class _FlipProbTargetOp(StimPrintable, IRDLOperation, ABC):
    """Gate op with qubit targets and an optional flip probability."""

    STIM_NAME: ClassVar[str]

    targets = prop_def(ArrayAttr[QubitAttr])
    flip_probability = opt_prop_def(FloatData)

    assembly_format = "(`(` $flip_probability^ `)`)? $targets attr-dict"

    def __init__(
        self,
        targets: ArrayAttr[QubitAttr] | Sequence[QubitAttr | int],
        flip_probability: float | FloatData | None = None,
    ):
        if not isinstance(targets, ArrayAttr):
            targets = ArrayAttr(
                QubitAttr(t) if isinstance(t, int) else t for t in targets
            )
        if isinstance(flip_probability, (int, float)):
            flip_probability = FloatData(flip_probability)
        super().__init__(
            properties={
                "targets": targets,
                "flip_probability": flip_probability,
            }
        )

    def print_stim(self, printer: StimPrinter) -> None:
        printer.print_string(self.STIM_NAME)
        if self.flip_probability is not None:
            printer.print_string(f"({self.flip_probability.data})")
        for target in self.targets:
            printer.print_string(" ")
            target.print_stim(printer)


class MeasurementOp(_FlipProbTargetOp, ABC):
    """Measurement of one or more qubits in some basis."""


@irdl_op_definition
class MOp(MeasurementOp):
    name = "stim.m"
    STIM_NAME: ClassVar[str] = "M"


@irdl_op_definition
class MXOp(MeasurementOp):
    name = "stim.mx"
    STIM_NAME: ClassVar[str] = "MX"


@irdl_op_definition
class MYOp(MeasurementOp):
    name = "stim.my"
    STIM_NAME: ClassVar[str] = "MY"


class ResetOp(StimPrintable, IRDLOperation, ABC):
    """
    Base class for reset operations.

    These operations reset one or more qubits to a specified basis state.
    """

    STIM_NAME: ClassVar[str]

    targets = prop_def(ArrayAttr[QubitAttr])

    assembly_format = "$targets attr-dict"

    def __init__(self, targets: ArrayAttr[QubitAttr] | Sequence[QubitAttr | int]):
        if not isinstance(targets, ArrayAttr):
            targets = ArrayAttr(
                QubitAttr(t) if isinstance(t, int) else t for t in targets
            )
        super().__init__(properties={"targets": targets})

    def print_stim(self, printer: StimPrinter) -> None:
        printer.print_string(self.STIM_NAME)
        for target in self.targets:
            printer.print_string(" ")
            target.print_stim(printer)


@irdl_op_definition
class ROp(ResetOp):
    name = "stim.r"
    STIM_NAME: ClassVar[str] = "R"


@irdl_op_definition
class RXOp(ResetOp):
    name = "stim.rx"
    STIM_NAME: ClassVar[str] = "RX"


@irdl_op_definition
class RYOp(ResetOp):
    name = "stim.ry"
    STIM_NAME: ClassVar[str] = "RY"


class MeasureResetOp(_FlipProbTargetOp, ABC):
    """Measurement followed by reset of one or more qubits."""


@irdl_op_definition
class MROp(MeasureResetOp):
    name = "stim.mr"
    STIM_NAME: ClassVar[str] = "MR"


@irdl_op_definition
class MRXOp(MeasureResetOp):
    name = "stim.mrx"
    STIM_NAME: ClassVar[str] = "MRX"


@irdl_op_definition
class MRYOp(MeasureResetOp):
    name = "stim.mry"
    STIM_NAME: ClassVar[str] = "MRY"
