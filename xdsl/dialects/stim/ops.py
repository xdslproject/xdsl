from abc import ABC
from collections.abc import Sequence
from io import StringIO

from xdsl.dialects.builtin import ArrayAttr, BoolAttr, FloatData, IntAttr, UnitAttr, i1
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
    var_result_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import OpTrait
from xdsl.utils.exceptions import ParseError, PyRDLAttrDefinitionError, PyRDLOpDefinitionError, VerifyException
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
            printer.print_list(self.coords, lambda attr: printer.print(attr.data))

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

    def print_parameter(self, printer: Printer) -> None:
        printer.print(" ")
        super().print_parameter(printer)

    def print_stim(self, printer: StimPrinter):
        printer.print_string(self.data)

# region Noise model attribute definitions

"""
In this implementation noise channels are attributes attached to other operations.

Providing noise as attributes allows them to be considered in the same way that other compiler analyses (like liveness parameters) are represented in xDSL and MLIR.

Stim provides these noise channel operations:
    1. Correlated Errors - correlated/e applies a pauli product with a given probability (has an else) + set of pauli and targets
    2. Depolarizing Errors
    3. Heralded - erase by doing half x half z. 1 paren for probability, targets. (tells you which qubits have been erased)
    4. heralded pauli - pauli modifiers with percent for eacg (can also be unheralded)
    5. pauli channel same but 2 qubits so up to 15
    5, x,y,z
    6. Measurement Errors
"""
class NoiseAttr(StimPrintable, ParametrizedAttribute, ABC):
    name = "stim.noiseattr"

@irdl_attr_definition
class DepolarizingNoiseAttr(NoiseAttr):
    """
        This attribute represents depolarizing noise occurring after the operation it is attached to.

        It can be attached to a single qubit operation or a two qubit operation, and infers its behaviour from which it can be attached to.

        This attribute takes one parameter `probability`, which is the probability of a depolarizing error occuring.

        In stim's simulator, this is then uniformly split into any of the possible combinations of pauli errors occuring.
        
        This also has an optional heralded operand which may only be set if the attached operation is single qubit (TODO: why can't this be attached to two qubit errors?)
    """
    name = "stim.depolarizingnoiseattr"
    
    probability : ParameterDef[FloatData]

    #TODO: implement heralding
    #heralded : ParameterDef[UnitAttr | NoneAttr]

    def __init__(self, probability:float):
        super().__init__(parameters = [FloatData(probability)])
    
    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        printer.print(self.probability.data)
        #if self.heralded is not NoneAttr:
        #    printer.print_string(", ")
        #    printer.print_string("heralded")
        printer.print_string(">")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        if parser.parse_optional_characters("<") is None:
            raise (PyRDLAttrDefinitionError("Probability attribute required!"))
        probability = parser.parse_number(context_msg=" probability parameter for depolarizing noise attribute")
       # if parser.parse_optional_characters(",") is None:
        parser.parse_characters(">", " for stim.depolarizingnoiseattr parameters")
        return [FloatData(probability)]#, NoneAttr()]
        #parser.parse_characters(",", " between stim.depolarizingnoiseattr args")
        #parser.parse_keyword("heralded", " as only possible second stim.depolarizingnoiseattr arg")
        #parser.parse_characters(">", " to end stim.depolarizingnoiseattr parameters")
        #return [FloatData(probability), UnitAttr()]

    def print_stim(self, printer:StimPrinter):
        with printer.in_parens():
            printer.print_attribute(self.probability)


@irdl_attr_definition
class SpecifiedPauliNoiseAttr(NoiseAttr):
    """
    This attribute represents non-uniform pauli errors occurring.

    It has a variadic parameter `probabilities`.

    If it is attached to a one qubit operation it must take 3 parameters, one indicating the probability of each pauli error occurring with interpreted format:
    (pX,pY,pZ)

    If it is attached to a two qubit operation it takes 15
    (pXI,pXX,pXY,pXZ,pYI,PYX,pYY,pYZ,pZI,pZX,pZY,pZZ,pIX,pIY,pIZ)

    Single pauli errors are also represented by this attribute, where they are encoded by being attached to a single qubit operation, with two of the paulis having 0 as their probability.

    TODO: Make this able to be given parameters which are specified by naming the errors and attributes for readability? - could do with a verifier? nah no difference

    For single qubit unitaries - a herald probability may also be passed.
    """

    name = "stim.specifiedpaulinoiseattr"

    probabilities: ParameterDef[ArrayAttr[FloatData]]

    def __init__(
        self,
        coords: list[float] | ArrayAttr[FloatData],
    ) -> None:
        if not isinstance(coords, ArrayAttr):
            coords = ArrayAttr(
                FloatData(arg)
                for arg in coords
            )
        super().__init__(parameters=[coords])

    def verify(self) -> None :
        raise NotImplementedError()

    @classmethod
    def parse_parameters(
        cls, parser: AttrParser
    ) -> Sequence[ArrayAttr[FloatData]]:
        probabilities = parser.parse_comma_separated_list(
            delimiter=parser.Delimiter.ANGLE,
            parse=lambda: FloatData(parser.parse_number(allow_boolean=False)),
        )
        return [ArrayAttr(probabilities)]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_list(self.probabilities, lambda attr: printer.print_attribute(attr))


    def print_stim(self, printer: StimPrinter):
        printer.print_attribute(self.probabilities)
        

@irdl_attr_definition
class PauliProductNoiseAttr(NoiseAttr):
    """
    This attribute represents a product of pauli errors occurring on multiple qubits. 

    It has one parameter for the probability of occurring, and another variadic attribute that specifies the pauli errors that occur and the qubits they occur on..
 
    We encode elses simply by the ordering of the appearance of any of these in an operations noise attribute.

    Technically all other errors are an example of these.
    """

    name = "stim.pauli_product_noiseattr"

    probability : ParameterDef[FloatData]

    paulis : ParameterDef[ArrayAttr[PauliAttr]]

    def verify_targets(self, indices:int) -> None:
        if len(self.paulis) != indices:
            raise PyRDLAttrDefinitionError("Pauli products must specify pairs of paulis and targets.")

    def print_stim(self, printer: StimPrinter) -> None: 
        printer.print_attribute(self.probability)


# endregion


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

    def print_parameter(self, printer: Printer) -> None:
        printer.print(" ")
        super().print_parameter(printer)


class TwoQubitGateAttr(EnumAttribute[TwoQubitCliffordsEnum]):
    name = "stim.twoqubitclifford"

    def print_parameter(self, printer: Printer) -> None:
        printer.print(" ")
        super().print_parameter(printer)


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

    noise = opt_prop_def(DepolarizingNoiseAttr)
    #TODO: Extend this to allow the alternatives - SpecifiedPauliNoiseAttrs, or an array of PauliProducts.

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
        noise: DepolarizingNoiseAttr | float | None = None,
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
        if noise is not None:
            if not isinstance(noise, DepolarizingNoiseAttr):
                noise = DepolarizingNoiseAttr(noise)
            properties["noise"] = noise
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
        
        noise = parser.parse_optional_number()
            
        targets = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_unresolved_operand
        )
        # parser.parse_punctuation(":")
        # qubits_type = parser.parse_type()
        # if qubits_type is not [qubit]:
        #    parser.raise_error("Targets must be a list of type !qref.qubit")
        qubits = parser.resolve_operands(targets, len(targets) * [qubit], parser.pos)

        return cls(gate_name, qubits, pauli_modifier, ctrl, dag, sqrt, noise)

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
        if self.noise is not None:
            printer.print(" ")
            printer.print(self.noise.probability.data)
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
                    if self.ctrl is not None or self.pauli_modifier is not None:
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

        if self.noise is not None:
            #TODO: When the other noise models are allowed this needs to be updated.
            printer.print_string("\n")
            printer.print_string("DEPOLARIZE")
            if isinstance(self.gate_name, SingleQubitGateAttr):
                printer.print_string("1")
            else:
                printer.print_string("2")
            self.noise.print_stim(printer)
            printer.print_targets(self.targets)
        


# endregion

# region Stabilizer operation definitions


@irdl_op_definition
class MeasurementGateOp(StimPrintable, GateOp, IRDLOperation):
    """
    Measurements take parens for noise.
    """

    name = "stim.measure"

    pauli_modifier = prop_def(PauliAttr)
    targets = var_operand_def(qubit)

    results = var_result_def(BoolAttr)

    noise = opt_prop_def(DepolarizingNoiseAttr)

    traits = frozenset([GateOpInterface()])

    def __init__(
        self,
        targets: Sequence[SSAValue],
        pauli_modifier: PauliOperatorEnum | PauliAttr = PauliOperatorEnum.Z,
        noise: DepolarizingNoiseAttr | float | None = None,
    ):
        if isinstance(pauli_modifier, PauliOperatorEnum):
            pauli_modifier = PauliAttr(pauli_modifier)
        if not isinstance(noise,DepolarizingNoiseAttr):
            if noise is not None:
                noise = DepolarizingNoiseAttr(noise)
        super().__init__(
            operands=[targets],
            result_types=[[i1] * len(targets)],
            properties={"pauli_modifier": pauli_modifier, "noise": noise},
        )

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print(self.pauli_modifier.data)
        if self.noise is not None:
            printer.print_string(" ")
            printer.print(self.noise.probability.data)
        printer.print_string(" ")
        printer.print_operands(self.targets)

    @classmethod
    def parse(cls, parser: Parser):
        pauli_modifier = parser.parse_str_enum(PauliOperatorEnum)
        properties:dict[str,Attribute] = {"pauli_modifier": PauliAttr(pauli_modifier)}
        noise_probability = parser.parse_optional_number()
        if noise_probability is not None:
            properties["noise"] = DepolarizingNoiseAttr(noise_probability)
        targets = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_unresolved_operand
        )
        qubits = parser.resolve_operands(targets, len(targets) * [qubit], parser.pos)
        return cls.create(
            operands=qubits,
            result_types=[i1] * len(qubits),
            properties=properties,
        )

    def print_stim(self, printer: StimPrinter):
        printer.print_string("M")
        printer.print_string(self.pauli_modifier.data)
        if self.noise is not None:
            self.noise.print_stim(printer)
        printer.update_ssa_results(self.results)
        printer.print_targets(self.targets)


@irdl_op_definition
class ResetGateOp(StimPrintable, GateOp, IRDLOperation):
    """
    Resets take no parens.
    """

    name = "stim.reset"

    pauli_modifier = prop_def(PauliAttr)
    targets = var_operand_def(qubit)

    traits = frozenset([GateOpInterface()])

    def __init__(
        self,
        targets: Sequence[SSAValue],
        pauli_modifier: PauliOperatorEnum | PauliAttr = PauliOperatorEnum.Z,
    ):
        if isinstance(pauli_modifier, PauliOperatorEnum):
            pauli_modifier = PauliAttr(pauli_modifier)
        super().__init__(
            operands=[targets],
            properties={"pauli_modifier": pauli_modifier},
        )

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print(self.pauli_modifier.data)
        printer.print_string(" ")
        printer.print_operands(self.targets)

    @classmethod
    def parse(cls, parser: Parser):
        pauli_modifier = parser.parse_str_enum(PauliOperatorEnum)
        targets = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_unresolved_operand
        )
        qubits = parser.resolve_operands(targets, len(targets) * [qubit], parser.pos)
        return cls.create(
            operands=qubits, properties={"pauli_modifier": PauliAttr(pauli_modifier)}
        )

    def print_stim(self, printer: StimPrinter):
        printer.print_string("R")
        printer.print_string(self.pauli_modifier.data)
        printer.print_targets(self.targets)


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

"""


@irdl_op_definition
class TickAnnotationOp(AnnotationOp):
    """
    A tick annotation is essentially an empty marker that can be used by the compiler.
    """

    name = "stim.tick"
    assembly_format = "attr-dict"

    def __init__(self):
        super().__init__(operands=[])

    def print_stim(self, printer: StimPrinter) -> None:
        printer.print_string("TICK")


# endregion

# region Controlflow operations

"""
@irdl_op_definition
class RepeatOp(scf.for)
"""

# endregion
