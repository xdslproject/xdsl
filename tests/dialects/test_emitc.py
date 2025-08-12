import pytest

from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import i32
from xdsl.dialects.emitc import EmitC_AddOp, EmitC_ArrayType, EmitC_CallOpaqueOp
from xdsl.utils.exceptions import VerifyException


def test_emitc_array_negative_dimension():
    with pytest.raises(
        VerifyException, match="EmitC array dimensions must have non-negative size"
    ):
        EmitC_ArrayType([-1], i32)


def test_call_opaque_with_str_callee():
    """
    Test that EmitC_CallOpaqueOp can be created with a string callee.
    """
    EmitC_CallOpaqueOp(callee="test", call_args=[], result_types=[])


def test_emitc_add_op_default_result_type():
    """
    Test that EmitC_AddOp uses lhs.type as default result_type when result_type is None.
    """
    lhs_op = ConstantOp.from_int_and_width(1, i32)
    rhs_op = ConstantOp.from_int_and_width(2, i32)

    # Create AddOp without specifying result_type
    add_op = EmitC_AddOp(lhs=lhs_op.result, rhs=rhs_op.result)

    # Verify that result type matches lhs type
    assert add_op.result.type == i32
    assert add_op.result.type == lhs_op.result.type
