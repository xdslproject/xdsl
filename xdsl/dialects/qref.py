from __future__ import annotations

from abc import ABC, abstractmethod

from xdsl.dialects import qssa
from xdsl.dialects.builtin import IntegerType
from xdsl.ir import Dialect, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    VarOpResult,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer


@irdl_attr_definition
class QubitAttr(ParametrizedAttribute, TypeAttribute):
    """
    Reference to a qubit
    """

    name = "qref.qubit"


qubit = QubitAttr()


class QRefBase(IRDLOperation, ABC):
    """
    Base class for qref operations, with methods to help convert to the qssa dialect.

    Invariant:
    self.is_gate == self.ssa_op().is_gate
    """

    @abstractmethod
    def ssa_op(self) -> qssa.QssaBase:
        """
        Build corresponding qssa operation
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_gate(self) -> bool:
        """
        Is this operation a gate?
        Qref gates represent standard quantum logic gates
        They should have no results
        The results of the generated gate must be rewired when converting to the qssa dialect
        """
        raise NotImplementedError()


@irdl_op_definition
class QRefAllocOp(QRefBase):
    name = "qref.alloc"

    def ssa_op(self) -> qssa.QubitAllocOp:
        return qssa.QubitAllocOp.create(
            result_types=[qssa.qubit] * self.num_qubits,
            attributes=self.attributes,
        )

    @property
    def is_gate(self) -> bool:
        return False

    res: VarOpResult = var_result_def(qubit)

    def __init__(self, num_qubits: int):
        super().__init__(
            operands=(),
            result_types=[[qubit] * num_qubits],
        )

    @property
    def num_qubits(self):
        return len(self.res)

    @classmethod
    def parse(cls, parser: Parser) -> QRefAllocOp:
        with parser.in_angle_brackets():
            num_qubits = parser.parse_integer()
        attr_dict = parser.parse_optional_attr_dict()
        return QRefAllocOp.create(
            result_types=[qubit] * num_qubits,
            attributes=attr_dict,
        )

    def print(self, printer: Printer):
        with printer.in_angle_brackets():
            printer.print(self.num_qubits)

        printer.print_op_attributes(self.attributes)


@irdl_op_definition
class HGateOp(QRefBase):
    name = "qref.h"

    def ssa_op(self) -> qssa.HGateOp:
        return qssa.HGateOp.create(
            operands=self.operands,
            result_types=(qssa.qubit,),
            attributes=self.attributes,
        )

    @property
    def is_gate(self) -> bool:
        return True

    input = operand_def(qubit)

    assembly_format = "$input attr-dict"

    def __init__(self, input: SSAValue):
        super().__init__(
            operands=(input,),
            result_types=(),
        )


@irdl_op_definition
class CNotGateOp(QRefBase):
    name = "qref.cnot"

    def ssa_op(self) -> qssa.CNotGateOp:
        return qssa.CNotGateOp.create(
            operands=self.operands,
            result_types=(qssa.qubit, qssa.qubit),
            attributes=self.attributes,
        )

    @property
    def is_gate(self) -> bool:
        return True

    in1 = operand_def(qubit)

    in2 = operand_def(qubit)

    assembly_format = "$in1 `,` $in2 attr-dict"

    def __init__(self, in1: SSAValue, in2: SSAValue):
        super().__init__(
            operands=(in1, in2),
            result_types=(qubit, qubit),
        )


@irdl_op_definition
class CZGateOp(QRefBase):
    name = "qref.cz"

    def ssa_op(self) -> qssa.CZGateOp:
        return qssa.CZGateOp.create(
            operands=self.operands,
            result_types=(qssa.qubit, qssa.qubit),
            attributes=self.attributes,
        )

    @property
    def is_gate(self) -> bool:
        return True

    in1 = operand_def(qubit)

    in2 = operand_def(qubit)

    assembly_format = "$in1 `,` $in2 attr-dict"

    def __init__(self, in1: SSAValue, in2: SSAValue):
        super().__init__(
            operands=(in1, in2),
            result_types=(),
        )


@irdl_op_definition
class MeasureOp(QRefBase):
    name = "qref.measure"

    def ssa_op(self) -> qssa.MeasureOp:
        return qssa.MeasureOp.create(
            operands=self.operands,
            result_types=(IntegerType(1),),
            attributes=self.attributes,
        )

    @property
    def is_gate(self) -> bool:
        return False

    input = operand_def(qubit)

    output = result_def(IntegerType(1))

    assembly_format = "$input attr-dict"

    def __init__(self, input: SSAValue):
        super().__init__(
            operands=[input],
            result_types=[IntegerType(1)],
        )


QREF = Dialect(
    "qref",
    [
        CNotGateOp,
        CZGateOp,
        HGateOp,
        MeasureOp,
        QRefAllocOp,
    ],
    [
        QubitAttr,
    ],
)
