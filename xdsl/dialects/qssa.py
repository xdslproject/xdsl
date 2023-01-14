# QSSA dialect
# operations:
#   - qssa.alloc             # DONE only op with side effects TODO: add support for dynamic types
#   - qssa.split             # DONE
#   - qssa.merge             # DONE
#   - qssa.dim               # DONE
#   - qssa.cast              # TODO: need to support dynamic qubit types first
#   - qssa.measure           # DONE
#   - qssa.gate, qssa.U      # for arbitrary gates. U is for a single qubit matrix (by describing angles)
#   - qssa.CNOT              # DONE
#   - qssa.{X, Y, Z}         # DONE
#   - qssa.{Rx, Ry, Rz}      # DONE
#   - qssa.S                 # DONE
#   - qssa.T                 # DONE
#   - qssa.H                 # DONE
#   - qssa.dag               # TODO: problem - this is actually an operation on gate operations => needs to map Gate Ops into "dagged" Gate Ops!
#       - two options for modelling this:
#           - 1. introduce new type "Gate" which is what gate ops will now return, and an "apply" operation to apply Gates on Qubits
#           - 2. leave as an identity operation on Qubits, but encode its semantics in the rewrite rules <-- using this for now
# attributes:
#   - qubit<int, or none (in which case is variadic)>. # this is a type of qubits
#   - angles (values between 0 and 2pi)                # this could be just a float to parameterise the qssa operation
# other dialects:
#   - scf, for scf.if and scf.for
#   - std, arithmetic and logical ops
# verifier algorithm for single use:
#   - qubit used at most once in the same region
#   - two use in different regions, neither region can be an ancestor of another (otherwise we have reuse)
#   - if qubit is within a for loop, i.e. scf.for { qubit }, then definition of qubit must be in the region too
# 
# TODO:
#   - dynamic qubit types
#   - add traits
#   - move params into operands rather than attributes
from __future__ import annotations

from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.ir import (
    Operation,
    SSAValue,
    Dialect,
    OpResult,
    ParametrizedAttribute,
    Attribute,
    Data,
)
from xdsl.irdl import (
    irdl_op_definition,
    irdl_attr_definition,
    ParameterDef,
    Operand,
    VarOperand,
    AttributeDef,
    BaseAttr,
    AttrConstraint,
    AnyOf,
    builder,
    attr_constr_coercion,
)
from xdsl.utils.exceptions import DiagnosticException

# other dialects
from xdsl.dialects.builtin import (
    IntAttr,
    IntegerType,
    i32,
    f64, f32, f16,
    Annotated,
    StringAttr,
    FloatAttr,
    TensorType,
    ArrayAttr
)

from typing import Union
from dataclasses import dataclass

# type attributes. Actually these are type functions: Qubits :: [Attribute] -> Attribute
@irdl_attr_definition
class Qubits(ParametrizedAttribute):
    """
    TODO: add support for dynamic qubit sizes
    """
    name = "qubits"

    # number of qubits
    n: ParameterDef[IntAttr]

    @staticmethod
    @builder
    def get_n_qubits_type(n_qubits: int) -> Qubits:
        return Qubits([IntAttr.from_int(n_qubits)])


q1_Type = Qubits.get_n_qubits_type(1)
q2_Type = Qubits.get_n_qubits_type(2)


# data attributes
@irdl_attr_definition
class Angle(Data[float]):
    name = "angle"

    @staticmethod
    def parse_parameter(parser: Parser) -> float:
        data = parser.parse_float_literal()
        return data

    @staticmethod
    def print_parameter(data: float, printer: Printer) -> None:
        printer.print_string(f"{data}")

    @staticmethod
    @builder
    def from_float(angle: float) -> Angle:
        return Angle(angle)


@irdl_op_definition
class Alloc(Operation):
    # TODO: support dynamic qubit types
    # create qubits of the form `|0^n>`
    name: str = "qssa.alloc"

    # attributes
    n_qubits = AttributeDef(BaseAttr(IntAttr))

    # no operands

    # output
    result: Annotated[OpResult, Qubits]

    @staticmethod
    def get(val: Union[int, Attribute]) -> Alloc:
        if isinstance(val, int):
            # convert to an IntAttr
            val = IntAttr.from_int(val)

        qubits_type = Qubits([val])
        return Alloc.build(result_types=[qubits_type], attributes={"n_qubits": val})


@irdl_op_definition
class Dim(Operation):
    name: str = "qssa.dim"

    # operands
    input_qubit: Annotated[Operand, Qubits]

    # output
    output_qubit: Annotated[OpResult, Qubits]
    dimension: Annotated[OpResult, i32]

    @staticmethod
    def apply(input_qubit: Union[Operation, SSAValue]) -> Dim:
        input_type = SSAValue.get(input_qubit).typ
        return Dim.build(operands=[input_qubit], result_types=[input_type, i32])


@irdl_op_definition
class Split(Operation):
    # split into 2 wires - TODO: split into variadic number of wires
    name: str = "qssa.split"

    # attributes
    # w1_qubits = AttributeDef(
    #     BaseAttr(IntAttr)
    # )  # qubits in wire 1. TODO: decide if this should be operand or attr?

    # operands
    w_in: Annotated[Operand, BaseAttr(Qubits)]
    w1_n_qubits: Annotated[Operand, BaseAttr(IntegerType)]

    # output
    w1_out: Annotated[OpResult, Qubits]
    w2_out: Annotated[OpResult, Qubits]

    @staticmethod
    def apply(
        w_in: Union[Operation, SSAValue], w1_out_n_qubits: Union[Operation, SSAValue]
    ) -> Split:
        # convert to an int
        w1_out_n_qubits_ssa = SSAValue.get(w1_out_n_qubits)
        attr_constr_coercion(IntegerType).verify(w1_out_n_qubits_ssa.typ)
        w1_out_n_qubits = w1_out_n_qubits.value.value.data
        
        # w1 should have type Qubits<w1_qubits.data>
        # w2 should have type Qubits<n - w1_qubits.data>, where type of input_qubis is Qubits<n>
        w_in_ssa = SSAValue.get(w_in)
        w_in_n_qubits = w_in_ssa.typ.n.data # TODO: handle the dynamic qubit case here
        w2_out_n_qubits = w_in_n_qubits - w1_out_n_qubits
        if w1_out_n_qubits <= 0 or w2_out_n_qubits <= 0:
            # TODO: use Qubit print method.
            # TODO: maybe move into verifier of Split
            raise DiagnosticException(
                f"Cannot split type {w_in_n_qubits} wire into type {(w1_out_n_qubits, )} wires"
            )

        w1_type = Qubits([IntAttr(w1_out_n_qubits)])
        w2_type = Qubits([IntAttr(w2_out_n_qubits)])
        return Split.build(
            operands=[w_in_ssa, w1_out_n_qubits_ssa],
            result_types=[w1_type, w2_type]
            # attributes={"w1_qubits": w1_qubits},
        )


@irdl_op_definition
class Merge(Operation):
    # merge several wires together
    name: str = "qssa.merge"

    # attributes

    # operands
    ws_in: Annotated[VarOperand, Qubits]

    # output
    w_out: Annotated[OpResult, Qubits]

    @staticmethod
    def _verify_and_get_n(w_in: Union[Operation, SSAValue]) -> int:
        w_in_typ = SSAValue.get(w_in).typ
        attr_constr_coercion(Qubits).verify(w_in_typ)
        return w_in_typ.n.data

    @staticmethod
    def apply(ws_in: list[Union[Operation, SSAValue]]) -> Merge:
        n_total = sum(Merge._verify_and_get_n(w_in) for w_in in ws_in)
        wn_type = Qubits([IntAttr(n_total)])
        return Merge.build(operands=[ws_in], result_types=[wn_type])


@irdl_op_definition
class Dag(Operation):
    name: str = "qssa.dag"

    # operands
    qubits_in: Annotated[Operand, Qubits]

    # output
    qubits_out: Annotated[OpResult, Qubits]

    @staticmethod
    def apply(input_qubits: Union[Operation, SSAValue]) -> Dag:
        return Dag.build(
            operands=[input_qubits], result_types=[SSAValue.get(input_qubits).typ]
        )


bit_type = IntegerType.from_width(1)


@irdl_op_definition
class Measure(Operation):
    name: str = "qssa.measure"

    # operands
    qubits_in: Annotated[Operand, Qubits]

    # output
    tensor_out: Annotated[OpResult, TensorType]

    @staticmethod
    def apply(input_qubits: Union[Operation, SSAValue]) -> Measure:
        # find the dimension of current qubits
        qubit_dim = SSAValue.get(input_qubits).typ.n.data
        bit_measure_type = TensorType.from_type_and_list(bit_type, [qubit_dim])
        return Measure.build(
            operands=[input_qubits], result_types=[bit_measure_type]
        )


@irdl_op_definition
class Cast(Operation):
    """
    Cast operation. Casts dynamic qubit types Qubits<?> into static ones Qubits<n> and vice versa
    """
    name: str = "qssa.cast"

    # operands
    qubits_in: Annotated[Operand, Qubits]

    # output
    qubits_out: Annotated[OpResult, Qubits]

    @staticmethod
    def apply(input_qubits: Union[Operation, SSAValue]) -> Cast:
        input_qubits_typ = SSAValue.get(input_qubits).typ
        output_qubits_typ = input_qubits_typ
        # for now, just return identity
        # TODO: actually implement the casting functionality for dynamic <-> static types

        return Cast.build(
            operands=[input_qubits], result_types=[input_qubits_typ]
        )


# class Gate(Operation, ABC):
#     @property
#     @abstractmethod
#     def name() -> str:
#         return "qssa.gate"
#
#     # attributes
#     dagged = OptAttributeDef(BaseAttr(DaggedAttr))


@dataclass
class SquareMatrixConstraint(AttrConstraint):
    # Check that an attribute satisfies the constraints
    def verify(self, attr: Attribute) -> None:
        attr_constr_coercion(ArrayAttr).verify(attr)

        # check to ensure array represents a square matrix
        matrix = attr.data
        if not matrix:
            raise DiagnosticException("Matrix cannot be empty.")

        # ensure array dimensions is 2^n for some n
        nrows = len(matrix)
        if not (nrows & (nrows-1) == 0):
            raise DiagnosticException(f"Matrix dimension ({nrows}x_) is not 2^n x 2^n")

        # ensure array is square
        first_row = matrix[0]
        if not all(len(row.data) == len(first_row.data) for row in matrix):
            raise DiagnosticException("Matrix is not square")


@irdl_op_definition
class MatrixGate(Operation):
    """
    Definition of a gate by a 2^n x 2^n matrix.
    Gates must be reversible thus square.
    """
    name: str = "qssa.gate"

    # attributes
    matrix = AttributeDef(SquareMatrixConstraint())

    # inputs
    input: Annotated[Operand, Qubits]

    # outputs
    out: Annotated[OpResult, Qubits]

    @staticmethod
    def _convert_to_array_attr(matrix: list[list[float]]) -> ArrayAttr:
        matrix_attr_elems = [[FloatAttr.from_value(float(e)) for e in row] for row in matrix]
        matrix_attr_rows = [ArrayAttr.from_list(row) for row in matrix_attr_elems]
        return ArrayAttr.from_list(matrix_attr_rows)

    @staticmethod
    def apply(input_qubits: Union[Operation, SSAValue], matrix: list[list[float]]) -> MatrixGate:
        qubit_type = SSAValue.get(input_qubits).typ
        # TODO: add check to ensure matrix is compatible with number of qubits, i.e. is 2^n x 2^n for qubit<n>
        matrix_attr = MatrixGate._convert_to_array_attr(matrix)
        return MatrixGate.build(operands=[input_qubits], result_types=[qubit_type], attributes={"matrix": matrix_attr})


@irdl_op_definition
class EulerUnitaryGate(Operation):
    """
    Single qubit unitary from Euler angles
    """
    name: str = "qssa.euler_gate"

    # inputs
    the: Annotated[Operand, AnyOf([f16, f32, f64])]
    phi: Annotated[Operand, AnyOf([f16, f32, f64])]
    lam: Annotated[Operand, AnyOf([f16, f32, f64])]
    input: Annotated[Operand, Qubits]

    # outputs
    out: Annotated[OpResult, Qubits]

    @staticmethod
    def apply(the: Union[Operation, SSAValue],
              phi: Union[Operation, SSAValue],
              lam: Union[Operation, SSAValue],
              input_qubits: Union[Operation, SSAValue]) -> EulerUnitaryGate:
        qubit_type = SSAValue.get(input_qubits).typ
        return EulerUnitaryGate.build(operands=[the, phi, lam, input_qubits], result_types=[qubit_type])


@irdl_op_definition
class CNOT(Operation):
    name: str = "qssa.cnot"

    # inputs
    input: Annotated[Operand, q2_Type]

    # outputs
    out: Annotated[OpResult, q2_Type]

    @staticmethod
    def apply(input_qubits: Union[Operation, SSAValue]) -> CNOT:
        return CNOT.build(operands=[input_qubits], result_types=[q2_Type])


@irdl_op_definition
class Pauli(Operation):
    name: str = "qssa.pauli"

    # attributes
    # TODO: add constraint on the "X", "Y", "Z" strs
    pauli_type = AttributeDef(BaseAttr(StringAttr))

    # inputs
    input: Annotated[Operand, q1_Type]

    # outputs
    out: Annotated[OpResult, q1_Type]

    @staticmethod
    def apply(input_qubits: Union[Operation, SSAValue], pauli_type: str) -> Pauli:
        return Pauli.build(
            operands=[input_qubits],
            result_types=[q1_Type],
            attributes={
                "pauli_type": StringAttr.from_str(pauli_type),
            },
        )


@irdl_op_definition
class PauliRotate(Operation):
    name: str = "qssa.pauli_rotate"

    # attributes
    # TODO: add constraint on the "Rx", "Ry", "Rz" strs
    # TODO: custom printing of the angles
    pauli_type = AttributeDef(BaseAttr(StringAttr))
    rotation = AttributeDef(BaseAttr(FloatAttr))  # in multiples of pi

    # inputs
    input: Annotated[Operand, q1_Type]

    # outputs
    out: Annotated[OpResult, q1_Type]

    @staticmethod
    def apply(
        input_qubits: Union[Operation, SSAValue], pauli_type: str, rotation: float = 0.0
    ) -> PauliRotate:
        return PauliRotate.build(
            operands=[input_qubits],
            result_types=[q1_Type],
            attributes={
                "pauli_type": StringAttr.from_str(pauli_type),
                "rotation": FloatAttr.from_value(rotation),
            },
        )


@irdl_op_definition
class SGate(Operation):
    name: str = "qssa.s"

    # inputs
    input: Annotated[Operand, q1_Type]

    # outputs
    out: Annotated[OpResult, q1_Type]

    @staticmethod
    def apply(input_qubit: Union[Operation, SSAValue]) -> SGate:
        return SGate.build(operands=[input_qubit], result_types=[q1_Type])


@irdl_op_definition
class HGate(Operation):
    name: str = "qssa.h"

    # inputs
    input: Annotated[Operand, q1_Type]

    # outputs
    out: Annotated[OpResult, q1_Type]

    @staticmethod
    def apply(input_qubit: Union[Operation, SSAValue]) -> HGate:
        return HGate.build(operands=[input_qubit], result_types=[q1_Type])


@irdl_op_definition
class TGate(Operation):
    name: str = "qssa.t"

    # inputs
    input: Annotated[Operand, q1_Type]

    # outputs
    out: Annotated[OpResult, q1_Type]

    @staticmethod
    def apply(input_qubit: Union[Operation, SSAValue]) -> TGate:
        return TGate.build(operands=[input_qubit], result_types=[q1_Type])


Quantum = Dialect([Alloc, CNOT, PauliRotate, Pauli, Merge, Split, SGate, HGate, TGate], [Qubits, Angle])