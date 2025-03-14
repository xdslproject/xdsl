from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir import SSAValue


def const_evaluate_operand_attribute(operand: SSAValue) -> IntegerAttr | None:
    """
    Try to constant evaluate an SSA value, returning None on failure.
    """
    if isinstance(op := operand.owner, arith.ConstantOp) and isinstance(
        val := op.value, IntegerAttr
    ):
        return val


def const_evaluate_operand(operand: SSAValue) -> int | None:
    """
    Try to constant evaluate an SSA value, returning None on failure.
    """
    if (attr := const_evaluate_operand_attribute(operand)) is not None:
        return attr.value.data
