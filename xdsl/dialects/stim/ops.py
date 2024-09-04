from abc import ABC
from collections.abc import Sequence
from io import StringIO

from xdsl.dialects.builtin import ArrayAttr, FloatData, IntAttr, UnitAttr
from xdsl.dialects.qref import qubit
from xdsl.dialects.stim.stim_printer import StimPrintable, StimPrinter
from xdsl.ir import (
    Attribute,
    EnumAttribute,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    ParameterDef,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    var_operand_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import OpTrait
from xdsl.utils.exceptions import PyRDLOpDefinitionError, VerifyException
from xdsl.utils.str_enum import StrEnum

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

    def __init__(self, body: Region):
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


# region Operation definitions

"""
Core Operations:
An operation is a quantum channel to apply to the quantum state of the system, and come under two groups:
    1. `Gate` operations - which are pure and can be applied by a system controlling a quantum computer
    2. `Stabilizer` operations - which are destructive to the quantum state.

The only gate operations currently supported by Stim are Clifford gates.

Stim splits stabilizer operations into measurement gates - which add data to the measurement record, and resets - which are silent and do not.
We model measurement results using SSA-values returned from operations here.

Stim also has `noise` operations which are used for simulating errors occurring. These are not part of the operational semantics that a control system
can implement, as they are purely for simulation and compilation purposes, so we model these as attached attributes to other operations.
Noise which occurs whilst doing nothing is then modelled by attaching an error attribute to an identity gate.
"""

"""
As Stim is built for error-correction, its basis is the pauli-operators, X, Y, and Z.
Most of its other operations can be considered as having a base operation modified by a Pauli indicating its basis.

"""


class PauliOperatorEnum(StrEnum):
    """
    Specify to explicitly indicate:
    * X : Pauli X operation
    * Y : Pauli Y operation
    * X : Pauli Z operation
    """

    X = "X"
    Y = "Y"
    Z = "Z"


class PauliAttr(StimPrintable, EnumAttribute[PauliOperatorEnum]):
    name = "stim.pauli"

    def print_stim(self, printer: StimPrinter):
        printer.print_string(self.data)


# region Gate Operation Enum definitions

"""
Gate operation definitions
"""


class SingleQubitCliffordsEnum(StrEnum):
    """
    Specify to explicitly indicate:
    * Rotation : Rotation gate - default is I but if modified by a Pauli, this gives a rotation about that axis.
    * BiAxisRotation : A rotation about two axes - the default is the Hadamard gate - modifiers give the axis not rotated about but this is made clear by printing.
    TODO: Currently do not support period cycling gate.
    """

    Rotation = "I"
    BiAxisRotation = "H"


class TwoQubitCliffordsEnum(StrEnum):
    """
    Specify to explicitly indicate:
    * Swap : A permutation operator that swaps the state of the two qubits it acts on
    * Controlled gate :
    * Midswap :
    """

    Swap = "Swap"
    Ctrl = "C_"
    Midswap = "Midswap"
    Both_Pauli = "Both_Pauli"


class SingleQubitGateAttr(EnumAttribute[SingleQubitCliffordsEnum]):
    name = "stim.singlequbitclifford"


class TwoQubitGateAttr(EnumAttribute[TwoQubitCliffordsEnum]):
    name = "stim.twoqubitclifford"


# endregion

# region Gate operation definitions


class GateOpInterface(OpTrait, ABC):
    """
    A gate operation always has:
        1. A gate set enum attribute indicating which gate it is.
        2. A variadic operand `targets` indicating what qubits it acts on.
        3. A parser and printer that checks the targets are formed correctly

    """

    def get_gate_name(self, op: Operation) -> Attribute:
        gate_name = op.get_attr_or_prop("gate_name")
        if gate_name is None or not isinstance(gate_name, EnumAttribute):
            raise VerifyException(
                f'Operation {op} must have a "gate_name" attribute of type '
                f"`EnumAttribute` to conform to {GateOpInterface.__name__}"
            )
        return gate_name


@irdl_op_definition
class GateOp(IRDLOperation, ABC):
    """
    Base gate operation for stim.
    """

    name = "stim.gate_op"

    gate_name = prop_def(Attribute)

    targets = operand_def(ArrayAttr[IntAttr])

    traits = frozenset([GateOpInterface()])

    def get_targets(self):
        return self.targets


@irdl_op_definition
class CliffordGateOp(StimPrintable, GateOp, IRDLOperation):
    """
    Clifford gates.
    """

    name = "stim.clifford"

    gate_name = prop_def(base(SingleQubitGateAttr) | base(TwoQubitGateAttr))
    pauli_modifier = opt_prop_def(PauliAttr)

    targets = var_operand_def(qubit)
    dag = opt_prop_def(UnitAttr, prop_name="dag")
    sqrt = opt_prop_def(UnitAttr, prop_name="sqrt")

    ctrl = opt_prop_def(PauliAttr)

    traits = frozenset([GateOpInterface()])

    def __init__(
        self,
        gate_name: (SingleQubitCliffordsEnum | TwoQubitCliffordsEnum)
        | (SingleQubitGateAttr | TwoQubitGateAttr),
        targets: Sequence[SSAValue],
        pauli_modifier: PauliOperatorEnum | PauliAttr | None = None,
        ctrl: PauliOperatorEnum | PauliAttr | None = None,
        is_dag: bool = False,
        is_sqrt: bool = False,
    ):
        properties: dict[str, Attribute] = {}
        if isinstance(gate_name, SingleQubitCliffordsEnum):
            gate_name = SingleQubitGateAttr(gate_name)
            if ctrl is not None:
                raise PyRDLOpDefinitionError("Single qubit gates cannot be controlled!")
        if isinstance(gate_name, TwoQubitCliffordsEnum):
            gate_name = TwoQubitGateAttr(gate_name)
            if isinstance(ctrl, PauliOperatorEnum):
                ctrl = PauliAttr(ctrl)
                properties["ctrl"] = ctrl
        properties["gate_name"] = gate_name
        if isinstance(pauli_modifier, PauliOperatorEnum):
            properties["pauli_modifier"] = PauliAttr(pauli_modifier)
        if isinstance(pauli_modifier, PauliAttr):
            properties["pauli_modifier"] = pauli_modifier
        if is_dag:
            properties["dag"] = UnitAttr()
        if is_sqrt:
            properties["sqrt"] = UnitAttr()
        super().__init__(operands=[targets], properties=properties)

    def verify_(self) -> None:
        if isinstance(self.gate_name, TwoQubitGateAttr):
            if len(self.targets) % 2 != 0:
                raise PyRDLOpDefinitionError(
                    "Two qubit gates expect an even number of targets."
                )

    @classmethod
    def parse(cls, parser: Parser):
        """
        Parse assembly without names of properties with the form:
            stim.clifford $gate_name $pauli_modifier? $dag? $sqrt? `(` $targets `)`

        """
        if (
            gate_name := parser.parse_optional_str_enum(SingleQubitCliffordsEnum)
        ) is None:
            if (
                gate_name := parser.parse_optional_str_enum(TwoQubitCliffordsEnum)
            ) is None:
                parser.raise_error(
                    "Expected a gate name of either SingleQubitGateAttr or TwoQubitGateAttr for stim.clifford."
                )
            else:
                gate_name = TwoQubitGateAttr(gate_name)
        else:
            gate_name = SingleQubitGateAttr(gate_name)

        pauli_modifier = parser.parse_optional_str_enum(PauliOperatorEnum)

        dag = parser.parse_optional_keyword("dag") is not None
        sqrt = parser.parse_optional_keyword("sqrt") is not None
        ctrl = parser.parse_optional_str_enum(PauliOperatorEnum)

        targets = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_unresolved_operand
        )
        # parser.parse_punctuation(":")
        # qubits_type = parser.parse_type()
        # if qubits_type is not [qubit]:
        #    parser.raise_error("Targets must be a list of type !qref.qubit")
        qubits = parser.resolve_operands(targets, len(targets) * [qubit], parser.pos)

        return cls(gate_name, qubits, pauli_modifier, ctrl, dag, sqrt)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print(self.gate_name.data)
        if self.pauli_modifier is not None:
            printer.print(" ")
            printer.print(self.pauli_modifier.data)
        if self.dag is not None:
            printer.print(" dag")
        if self.sqrt is not None:
            printer.print(" sqrt")
        if self.ctrl is not None:
            printer.print(" ")
            printer.print(self.ctrl.data)
        printer.print(" ")
        printer.print_operands(self.targets)

    def print_stim(self, printer: StimPrinter):
        if isinstance(self.gate_name.data, SingleQubitCliffordsEnum):
            match self.gate_name.data:
                case SingleQubitCliffordsEnum.Rotation:
                    if self.sqrt is not None:
                        if (
                            self.pauli_modifier is None
                            or self.pauli_modifier.data == PauliOperatorEnum.Z
                        ):
                            printer.print_string("S")
                        else:
                            printer.print_string("SQRT_")
                            self.pauli_modifier.print_stim(printer)
                        if self.dag is not None:
                            printer.print_string("_DAG")
                    elif self.pauli_modifier is not None:
                        self.pauli_modifier.print_stim(printer)
                    else:
                        printer.print_string("I")
                case SingleQubitCliffordsEnum.BiAxisRotation:
                    if self.sqrt is not None:
                        raise ValueError(
                            f"Sqrt of BiAxisRotation gate ({self}) is not supported by Stim."
                        )
                    match self.pauli_modifier:
                        case PauliAttr(PauliOperatorEnum.Y) | None:
                            printer.print_string("H")
                        case PauliAttr(PauliOperatorEnum.X):
                            printer.print_string("H_YZ")
                        case _:
                            printer.print_string("H_XY")
        else:
            match self.gate_name.data:
                case TwoQubitCliffordsEnum.Swap:
                    if self.ctrl != None or self.pauli_modifier != None:
                        raise ValueError("Controlled or modified swaps not supported.")
                    printer.print_string("SWAP")
                case TwoQubitCliffordsEnum.Ctrl:
                    if (
                        self.pauli_modifier is not None
                        and self.pauli_modifier.data != PauliOperatorEnum.Z
                    ):
                        self.pauli_modifier.print_stim(printer)
                    printer.print_string("C")
                    if self.ctrl is None or self.ctrl.data == PauliOperatorEnum.X:
                        printer.print_string("NOT")
                    else:
                        self.ctrl.print_stim(printer)
                case TwoQubitCliffordsEnum.Midswap:
                    dag = False
                    if self.ctrl is not None:
                        raise ValueError("Controlled midwaps not supported")
                    if self.pauli_modifier is not None:
                        if self.pauli_modifier.data == PauliOperatorEnum.Y:
                            printer.print_string("I")
                            if self.dag is not None:
                                dag = True
                        else:
                            printer.print_string("C")
                            self.pauli_modifier.print_stim(printer)
                    else:
                        printer.print_string("I")
                        if self.dag is not None:
                            dag = True
                    printer.print_string("SWAP")
                    if dag:
                        printer.print_string("DAG")
                case TwoQubitCliffordsEnum.Both_Pauli:
                    if self.ctrl is not None:
                        raise ValueError(
                            "Controlled multi rotation gates are not defined in Stim"
                        )
                    if self.sqrt is not None:
                        raise ValueError(
                            "This constructor is intended only for SQRT paulis on pairs of input qubits. Use single qubit cliffords instead."
                        )
                    printer.print_string("SQRT_")
                    if self.pauli_modifier is None:
                        # default the sqrt to XX
                        printer.print_string("XX")
                    else:
                        self.pauli_modifier.print_stim(printer)
                        self.pauli_modifier.print_stim(printer)
                    if self.dag is not None:
                        printer.print_string("_DAG")

        printer.print_targets(self.targets)


# endregion

# region Stabilizer operation definitions


# endregion

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
