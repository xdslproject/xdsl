from xdsl.ir import Operation, Dialect, Attribute
from xdsl.irdl import IRDLOperation, irdl_op_definition, var_operand_def, operand_def, result_def
from collections.abc import Sequence

@irdl_op_definition
class AffineSelectOp(IRDLOperation):
    name = "hida_prim.affine.select"

    args = var_operand_def()
    false_value = operand_def()
    true_value = operand_def()
    res = result_def()

    def __init__(self, args: Sequence[Operation], false_value: Operation, true_value: Operation, res_type : Attribute):
        super().__init__(operands=list(args)+[false_value, true_value], result_types=[res_type])

HIDA_prim = Dialect(
    "hida_prim",
    [
        AffineSelectOp
    ],
    []
)