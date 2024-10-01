from collections.abc import Sequence
from io import StringIO

from xdsl.dialects.builtin import ArrayAttr, IntAttr
from xdsl.dialects.stim.stim_printer_parser import StimPrintable, StimPrinter
from xdsl.ir import ParametrizedAttribute, Region, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    ParameterDef,
    irdl_attr_definition,
    irdl_op_definition,
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

    qubit: ParameterDef[IntAttr]

    def __init__(self, qubit: int | IntAttr) -> None:
        if not isinstance(qubit, IntAttr):
            qubit = IntAttr(qubit)
        super().__init__(parameters=[qubit])

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[IntAttr]:
        with parser.in_angle_brackets():
            qubit = parser.parse_integer(allow_negative=False, allow_boolean=False)
            return (IntAttr(qubit),)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print(self.qubit.data)

    def print_stim(self, printer: StimPrinter):
        printer.print_string(f"{self.qubit.data}")


@irdl_attr_definition
class QubitMappingAttr(StimPrintable, ParametrizedAttribute):
    """
    This attribute provides a way to indicate the required connectivity or layout of `physical` qubits.

    It consists of two parameters:
        1. A co-ordinate array (currently it only anticipates a pair of qubits, but this is not fixed)
        2. A value associated with a qubit referred to in a circuit.

    The co-ordinates may be used as a physical address of a qubit, or the relative address with respect to some known physical address.

    Operations that attach this as a property may represent the lattice-like structure of a physical quantum computer by having a property with an ArrayAttr[QubitCoordsAttr].
    """

    name = "stim.qubit_coord"

    coords: ParameterDef[ArrayAttr[IntAttr]]
    qubit_name: ParameterDef[QubitAttr]

    def __init__(
        self, coords: list[int] | ArrayAttr[IntAttr], qubit_name: int | QubitAttr
    ) -> None:
        if not isinstance(qubit_name, QubitAttr):
            qubit_name = QubitAttr(qubit_name)
        if not isinstance(coords, ArrayAttr):
            coords = ArrayAttr(IntAttr(c) for c in coords)
        super().__init__(parameters=[coords, qubit_name])

    @classmethod
    def parse_parameters(
        cls, parser: AttrParser
    ) -> tuple[ArrayAttr[IntAttr], QubitAttr]:
        parser.parse_punctuation("<")
        coords = parser.parse_comma_separated_list(
            delimiter=parser.Delimiter.PAREN,
            parse=lambda: IntAttr(parser.parse_integer(allow_boolean=False)),
        )
        parser.parse_punctuation(",")
        qubit = parser.parse_attribute()
        if not isinstance(qubit, QubitAttr):
            parser.raise_error("Expected qubit attr", at_position=parser.pos)
        parser.parse_punctuation(">")
        return (ArrayAttr(coords), qubit)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print("(")
            for i, elem in enumerate(self.coords):
                if i:
                    printer.print_string(", ")
                printer.print(elem.data)
            printer.print("), ")
            printer.print(self.qubit_name)

    def print_stim(self, printer: StimPrinter):
        printer.print_attribute(self.coords)
        printer.print_string(" ")
        self.qubit_name.print_stim(printer)


@irdl_op_definition
class StimCircuitOp(StimPrintable, IRDLOperation):
    """
    Base operation containing a stim program
    """

    name = "stim.circuit"

    body = region_def("single_block")

    assembly_format = "attr-dict-with-keyword $body"

    def __init__(self, body: Region):
        super().__init__(regions=[body])

    def verify(self, verify_nested_ops: bool = True) -> None:
        return

    def print_stim(self, printer: StimPrinter):
        for op in self.body.block.ops:
            if not isinstance(op, StimPrintable):
                raise ValueError(f"Cannot print in stim format: {op}")
            op.print_stim(printer)
        printer.print_string("")

    def stim(self) -> str:
        io = StringIO()
        printer = StimPrinter(io)
        self.print_stim(printer)
        res = io.getvalue()
        return res
