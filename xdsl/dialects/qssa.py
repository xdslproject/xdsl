from __future__ import annotations

from abc import ABC, abstractmethod

from xdsl.dialects import qref
from xdsl.dialects.builtin import IntegerType
from xdsl.dialects.quantum import AngleAttr
from xdsl.ir import Dialect, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer


@irdl_attr_definition
class QubitAttr(ParametrizedAttribute, TypeAttribute):
    """
    Type for a single qubit.
    """

    name = "qssa.qubit"


qubit = QubitAttr()


class QssaBase(IRDLOperation, ABC):
    """
    Base class for qssa operations, with methods to help convert to the qref dialect.

    Invariant:
    self.is_gate == self.ref_op().is_gate
    """

    @abstractmethod
    def ref_op(self) -> qref.QRefBase:
        """
        Build corresponding qref operation.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_gate(self) -> bool:
        """
        Is this operation a gate?
        Qssa gates represent standard quantum logic gates.
        They should have an equal number of qubit-typed operands and results.
        Their results must be rewired when converting to the qref dialect.
        """
        raise NotImplementedError()


@irdl_op_definition
class QubitAllocOp(QssaBase):
    name = "qssa.alloc"

    res = var_result_def(qubit)

    def __init__(self, num_qubits: int):
        super().__init__(
            operands=(),
            result_types=[[qubit] * num_qubits],
        )

    @property
    def num_qubits(self):
        return len(self.res)

    @classmethod
    def parse(cls, parser: Parser) -> QubitAllocOp:
        with parser.in_angle_brackets():
            num_qubits = parser.parse_integer()
        attr_dict = parser.parse_optional_attr_dict()
        return QubitAllocOp.create(
            result_types=[qubit] * num_qubits,
            attributes=attr_dict,
        )

    def print(self, printer: Printer):
        with printer.in_angle_brackets():
            printer.print(self.num_qubits)

        printer.print_op_attributes(self.attributes)

    def ref_op(self) -> qref.QRefAllocOp:
        return qref.QRefAllocOp.create(
            result_types=[qref.qubit] * self.num_qubits,
            attributes=self.attributes,
        )

    @property
    def is_gate(self) -> bool:
        return False


@irdl_op_definition
class HGateOp(QssaBase):
    name = "qssa.h"

    input = operand_def(qubit)

    output = result_def(qubit)

    assembly_format = "$input attr-dict"

    def __init__(self, input: SSAValue):
        super().__init__(
            operands=(input,),
            result_types=(qubit,),
        )

    def ref_op(self) -> qref.HGateOp:
        return qref.HGateOp.create(
            operands=self.operands,
            result_types=(),
            attributes=self.attributes,
        )

    @property
    def is_gate(self) -> bool:
        return True


@irdl_op_definition
class CNotGateOp(QssaBase):
    name = "qssa.cnot"

    in1 = operand_def(qubit)

    in2 = operand_def(qubit)

    out1 = result_def(qubit)

    out2 = result_def(qubit)

    assembly_format = "$in1 `,` $in2 attr-dict"

    def __init__(self, in1: SSAValue, in2: SSAValue):
        super().__init__(
            operands=(in1, in2),
            result_types=(qubit, qubit),
        )

    def ref_op(self) -> qref.CNotGateOp:
        return qref.CNotGateOp.create(
            operands=self.operands,
            result_types=(),
            attributes=self.attributes,
        )

    @property
    def is_gate(self) -> bool:
        return True


@irdl_op_definition
class RZGateOp(QssaBase):
    name = "qssa.rz"

    input = operand_def(qubit)

    output = result_def(qubit)

    angle = prop_def(AngleAttr)

    assembly_format = "$angle $input attr-dict"

    def __init__(self, angle: AngleAttr, input: SSAValue):
        super().__init__(
            operands=(input,),
            result_types=(qubit,),
            properties={"angle": angle},
        )

    def ref_op(self) -> qref.RZGateOp:
        return qref.RZGateOp.create(
            operands=self.operands,
            result_types=(),
            attributes=self.attributes,
            properties=self.properties,
        )

    @property
    def is_gate(self) -> bool:
        return True


@irdl_op_definition
class MeasureOp(QssaBase):
    name = "qssa.measure"

    input = operand_def(qubit)

    output = result_def(IntegerType(1))

    assembly_format = "$input attr-dict"

    def __init__(self, input: SSAValue):
        super().__init__(
            operands=[input],
            result_types=[IntegerType(1)],
        )

    def ref_op(self) -> qref.MeasureOp:
        return qref.MeasureOp.create(
            operands=self.operands,
            result_types=(IntegerType(1),),
            attributes=self.attributes,
        )

    @property
    def is_gate(self) -> bool:
        return False


QSSA = Dialect(
    "qssa",
    [
        CNotGateOp,
        RZGateOp,
        HGateOp,
        MeasureOp,
        QubitAllocOp,
    ],
    [
        QubitAttr,
    ],
)
