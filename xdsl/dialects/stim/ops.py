from abc import ABC
from collections.abc import Sequence
from io import StringIO

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
            parse=lambda: IntAttr(x)
            if type(x := parser.parse_number(allow_boolean=False)) is int
            else FloatData(x),
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
                    lambda c: printer.print_int(c.data)
                    if isinstance(c, IntAttr)
                    else printer.print_float(c.data, f64),
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
