import pytest

from xdsl.dialects.builtin import (
    FloatAttr,
    IntegerAttr,
    SymbolRefAttr,
    f32,
    f64,
    i32,
    i64,
    i128,
)
from xdsl.dialects.wasmssa import (
    ConstantExpressionOpTrait,
    ConstOp,
    ExternRefType,
    FuncRefType,
    GlobalGetOp,
    ValType,
)
from xdsl.utils.exceptions import VerifyException


def test_constant_expression_ops_have_trait():
    assert ConstOp(IntegerAttr(1, i32)).has_trait(ConstantExpressionOpTrait)
    assert GlobalGetOp("global", i32).has_trait(ConstantExpressionOpTrait)


@pytest.mark.parametrize(
    "value",
    [
        IntegerAttr(1, i32),
        IntegerAttr(2, i64),
        FloatAttr(3.0, f32),
        FloatAttr(4.0, f64),
    ],
)
def test_const_op(value: IntegerAttr | FloatAttr):
    op = ConstOp(value)

    assert op.value == value
    assert op.result.type == value.get_type()
    op.verify()


def test_const_op_rejects_non_numeric_type():
    op = ConstOp.create(
        properties={"value": IntegerAttr(1, i128)},
        result_types=[i128],
    )

    with pytest.raises(VerifyException):
        op.verify()


def test_const_op_rejects_mismatched_value_and_result_types():
    op = ConstOp.create(
        properties={"value": IntegerAttr(1, i32)},
        result_types=[i64],
    )

    with pytest.raises(VerifyException):
        op.verify()


@pytest.mark.parametrize(
    "result_type",
    [i32, i64, i128, f32, f64, FuncRefType(), ExternRefType()],
)
@pytest.mark.parametrize("global_", ["global", SymbolRefAttr("global")])
def test_global_get_op(
    global_: str | SymbolRefAttr,
    result_type: ValType,
):
    op = GlobalGetOp(global_, result_type)

    assert op.global_ == SymbolRefAttr("global")
    assert op.global_val.type == result_type
    op.verify()


def test_global_get_op_rejects_nested_symbol_ref():
    op = GlobalGetOp.create(
        properties={"global": SymbolRefAttr("module", ["global"])},
        result_types=[i32],
    )

    with pytest.raises(VerifyException):
        op.verify()
