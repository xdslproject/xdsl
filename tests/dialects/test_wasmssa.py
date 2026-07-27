import pytest

from xdsl.dialects.builtin import f32, f64, i32, i64, i128
from xdsl.dialects.wasmssa import AddOp
from xdsl.ir import Attribute
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


@pytest.mark.parametrize("value_type", [i32, i64, f32, f64])
def test_add_op(value_type: Attribute):
    lhs = create_ssa_value(value_type)
    rhs = create_ssa_value(value_type)

    op = AddOp(lhs, rhs)

    assert op.lhs is lhs
    assert op.rhs is rhs
    assert op.result.type == value_type
    op.verify()


def test_add_op_rejects_non_numeric_type():
    lhs = create_ssa_value(i128)
    rhs = create_ssa_value(i128)
    op = AddOp(lhs, rhs)

    with pytest.raises(VerifyException):
        op.verify()


def test_add_op_rejects_mismatched_operand_types():
    lhs = create_ssa_value(i32)
    rhs = create_ssa_value(i64)
    op = AddOp(lhs, rhs)

    with pytest.raises(VerifyException):
        op.verify()
