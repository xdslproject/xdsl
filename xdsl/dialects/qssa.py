from xdsl.dialects.builtin import IntegerType
from xdsl.ir import Dialect, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)


@irdl_attr_definition
class QubitAttr(ParametrizedAttribute, TypeAttribute):
    """
    Type for a single qubit
    """

    name = "qssa.qubit"


qubit = QubitAttr()


@irdl_op_definition
class HGateOp(IRDLOperation):
    name = "qssa.h"

    input = operand_def(qubit)

    output = result_def(qubit)

    assembly_format = "$input attr-dict"

    def __init__(self, input: SSAValue):
        super().__init__(operands=(input,), result_types=(qubit,))


@irdl_op_definition
class CNotGateOp(IRDLOperation):
    name = "qssa.cnot"

    in1 = operand_def(qubit)

    in2 = operand_def(qubit)

    out1 = result_def(qubit)

    out2 = result_def(qubit)

    assembly_format = "$in1 $in2 attr-dict"

    def __init__(self, in1: SSAValue, in2: SSAValue):
        super().__init__(operands=(in1, in2), result_types=(qubit, qubit))


@irdl_op_definition
class CZGateOp(IRDLOperation):
    name = "qssa.cz"

    in1 = operand_def(qubit)

    in2 = operand_def(qubit)

    out1 = result_def(qubit)

    out2 = result_def(qubit)

    assembly_format = "$in1 $in2 attr-dict"

    def __init__(self, in1: SSAValue, in2: SSAValue):
        super().__init__(operands=(in1, in2), result_types=(qubit, qubit))


@irdl_op_definition
class MeasureOp(IRDLOperation):
    name = "qssa.measure"

    input = operand_def(qubit)

    output = result_def(IntegerType(1))

    assembly_format = "$input attr-dict"

    def __init__(self, input: SSAValue):
        super().__init__(operands=(input,), result_types=(IntegerType(1),))


QSSA = Dialect(
    "qssa",
    [
        HGateOp,
        CZGateOp,
        CNotGateOp,
        MeasureOp,
    ],
    [
        QubitAttr,
    ],
)
