from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir import SSAValue


def const_evaluate_operand(operand: SSAValue) -> int | None:
    """
    Try to constant evaluate an SSA value, returning None on failure.
    """
    if isinstance(op := operand.owner, arith.Constant) and isinstance(
        val := op.value, IntegerAttr
    ):
        return val.value.data
