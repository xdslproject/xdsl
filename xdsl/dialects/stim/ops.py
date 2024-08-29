from abc import ABC
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
    prop_def,
    region_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer


class StimAttr(StimPrintable):
    "Base Stim attribute"

    name = "stim.attr"


@irdl_attr_definition
class QubitAttr(StimAttr, ParametrizedAttribute, TypeAttribute):
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
        parser.parse_punctuation("<")
        qubit = parser.parse_integer(allow_negative=False, allow_boolean=False)
        parser.parse_punctuation(">")
        return [IntAttr(qubit)]

    def print_stim(self, printer: StimPrinter):
        printer.print_string(self.qubit.data.__str__())


@irdl_attr_definition
class QubitMappingAttr(StimAttr, ParametrizedAttribute):
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
            parse=lambda: IntAttr(parser.parse_integer()),
        )
        parser.parse_punctuation(",")
        qubit = parser.parse_integer(allow_negative=False, allow_boolean=False)
        parser.parse_punctuation(">")
        return (ArrayAttr(coords), QubitAttr(qubit))

    def print(self, printer: Printer) -> None:
        printer.print("(")
        printer.print_attribute(self.coords)
        printer.print(") ")
        printer.print(self.qubit_name)

    def print_stim(self, printer: StimPrinter):
        printer.print_attribute(self.coords)
        self.qubit_name.print_stim(printer)


class StimOp(StimPrintable, IRDLOperation, ABC):
    """
    Base Stim operation
    """

    ...


@irdl_op_definition
class StimCircuitOp(StimOp):
    """
    Base operation containing a stim program
    """

    name = "stim.circuit"

    body = region_def("single_block")

    # qubitlayout = opt_prop_def(ArrayAttr[QubitMappingAttr])

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


"""
Annotation Operations

Stim contains a number of `Annotations` - instructions which do not affect the operational semantics of the stim circuit -
but may provide useful information about the circuit being run or how decoding should be done.

These are essentially code-directives for compiler analyses on the circuit.

Here each is attached as an attribute instead - but as they may appear in the code, are also given operations that can
drive the change of a value and be used to direct printing of stim circuits.
"""


class AnnotationOp(StimOp, ABC):
    """
    Base Annotation operation
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
