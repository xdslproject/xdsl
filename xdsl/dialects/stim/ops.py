from abc import ABC
from collections.abc import Sequence
from io import StringIO

from xdsl.dialects.builtin import ArrayAttr, FloatData, IntAttr
from xdsl.dialects.qref import qubit
from xdsl.dialects.stim.stim_printer import StimPrintable, StimPrinter
from xdsl.ir import ParametrizedAttribute, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    ParameterDef,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer

# region Attribute definitions


@irdl_attr_definition
class QubitMappingAttr(StimPrintable, ParametrizedAttribute):
    """
    This attribute provides a way to indicate the required connectivity or layout of `physical` qubits.

    It has one parameter that represents a co-ordinate array

    It is attached to a SSA-value associated with a qubit referred to in a circuit.
    This value is allocated at definition in a QubitCoord op.

    The co-ordinates may be used as a physical address of a qubit, or the relative address with respect to some known physical address.

    Operations that attach this as a property may represent the lattice-like structure of a physical quantum computer by having a property with an ArrayAttr[QubitCoordsAttr].
    """

    name = "stim.qubit_coord"

    coords: ParameterDef[ArrayAttr[FloatData | IntAttr]]

    def __init__(
        self,
        coords: list[float] | ArrayAttr[FloatData | IntAttr],
    ) -> None:
        if not isinstance(coords, ArrayAttr):
            coords = ArrayAttr(
                (IntAttr(int(arg))) if (type(arg) is int) else (FloatData(arg))
                for arg in coords
            )
        super().__init__(parameters=[coords])

    @classmethod
    def parse_parameters(
        cls, parser: AttrParser
    ) -> Sequence[ArrayAttr[FloatData | IntAttr]]:
        coords = parser.parse_comma_separated_list(
            delimiter=parser.Delimiter.ANGLE,
            parse=lambda: IntAttr(x)
            if type(x := parser.parse_number(allow_boolean=False)) is int
            else FloatData(x),
        )
        return [ArrayAttr(coords)]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            for i, elem in enumerate(self.coords):
                if i:
                    printer.print_string(", ")
                printer.print(elem.data)

    def print_stim(self, printer: StimPrinter):
        printer.print_attribute(self.coords)


# endregion

# region Instruction definitions

"""
Stim Instructions Definitions

Stim divides these into three groups:
    1. Operations
    2. Annotations
    3. Control Flow

A Stim Circuit is represented by a StimCircuitOp,
which contains a single list of Stim instructions as a single region with a single block.
A StimCircuitOp has attributes that can act as storage for the annotations that occur in a Stim circuit.
"""


@irdl_op_definition
class StimCircuitOp(StimPrintable, IRDLOperation):
    """
    Base operation containing a stim program
    """

    name = "stim.circuit"

    body = region_def("single_block")

    assembly_format = "attr-dict-with-keyword $body"

    def __init__(self, body: Region, qubitlayout: None | ArrayAttr[QubitMappingAttr]):
        super().__init__(regions=[body])

    def verify(self, verify_nested_ops: bool = True) -> None:
        return

    def print_stim(self, printer: StimPrinter):
        for op in self.body.block.ops:
            if printer.print_op(op):
                printer.print_string("\n")
        printer.print_string("")

    def stim(self) -> str:
        io = StringIO()
        printer = StimPrinter(io)
        self.print_stim(printer)
        res = io.getvalue()
        return res


# endregion


# region Annotation operations

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

    ...


@irdl_op_definition
class QubitCoordsOp(AnnotationOp):
    """
    Annotation operation that assigns a qubit reference to a coordinate.
    """

    name = "stim.assign_qubit_coord"

    qubitcoord = prop_def(QubitMappingAttr)
    target = operand_def(qubit)

    assembly_format = "$qubitcoord $target attr-dict"

    def __init__(self, target: SSAValue, qubitmapping: QubitMappingAttr):
        super().__init__(operands=[target], properties={"qubitcoord": qubitmapping})

    def print_stim(self, printer: StimPrinter) -> None:
        printer.print_string("QUBIT_COORDS")
        self.qubitcoord.print_stim(printer)
        printer.print_targets([self.target])


"""
@irdl_op_definition
class DetectorAnnotation(AnnotationOp):

class MPadAnnotation(AnnotationOp):

class ObservableAnnotation():

class ShiftCoordsAnnotation():

class TickAnnotation():
"""
# endregion

# region Controlflow operations

"""
@irdl_op_definition
class RepeatOp(scf.for)
"""

# endregion
