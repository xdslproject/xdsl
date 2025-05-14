import pytest

from xdsl.dialects.builtin import IntegerAttr
from xdsl.dialects.smt import (
    AndOp,
    BoolType,
    ConstantBoolOp,
    DistinctOp,
    EqOp,
    OrOp,
    VariadicBoolOp,
    XOrOp,
)
from xdsl.utils.test_value import create_ssa_value


def test_constant_bool():
    op = ConstantBoolOp(True)
    assert op.value is True
    assert op.value_attr == IntegerAttr(-1, 1)

    op = ConstantBoolOp(False)
    assert op.value is False
    assert op.value_attr == IntegerAttr(0, 1)


@pytest.mark.parametrize("op_type", [AndOp, OrOp, XOrOp, EqOp, DistinctOp])
def test_variadic_bool_op(op_type: type[VariadicBoolOp]):
    arg1 = create_ssa_value(BoolType())
    arg2 = create_ssa_value(BoolType())
    arg3 = create_ssa_value(BoolType())
    op = op_type(arg1, arg2, arg3)
    assert op.result.type == BoolType()
    assert list(op.inputs) == [arg1, arg2, arg3]
