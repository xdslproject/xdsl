import pytest

import xdsl.frontend.pyast.dialects.builtin as builtin
from xdsl.dialects.arith import AddiOp, ConstantOp, MulfOp
from xdsl.dialects.builtin import (
    Float32Type,
    Float64Type,
    FloatAttr,
    IntegerType,
    f32,
    f64,
    i32,
    i64,
)
from xdsl.frontend.pyast.exception import FrontendProgramException
from xdsl.frontend.pyast.op_resolver import OpResolver


def test_raises_exception_on_unknown_op():
    with pytest.raises(FrontendProgramException) as err:
        _ = OpResolver.resolve_op("xdsl.frontend.pyast.dialects.arith", "unknown")
    assert (
        err.value.msg
        == "Internal failure: operation 'unknown' does not exist in module 'xdsl.frontend.pyast.dialects.arith'."
    )


def test_raises_exception_on_unknown_overload():
    with pytest.raises(FrontendProgramException) as err:
        _ = OpResolver.resolve_op_overload("__unknown__", builtin._Integer)
    assert (
        err.value.msg == "Internal failure: '_Integer' does not overload '__unknown__'."
    )


def test_resolves_ops():
    a = ConstantOp.from_int_and_width(1, i32)
    b = ConstantOp.from_int_and_width(2, i32)

    addi = OpResolver.resolve_op("xdsl.frontend.pyast.dialects.arith", "addi")
    addi_op = addi(a, b)

    assert isinstance(addi_op, AddiOp)
    assert addi_op.operands[0] == a.results[0]
    assert addi_op.operands[1] == b.results[0]
    assert isinstance(addi_op.results[0].type, IntegerType)
    assert addi_op.results[0].type.width.data == 32

    c = ConstantOp(FloatAttr(5.0, f32))
    mulf = OpResolver.resolve_op("xdsl.frontend.pyast.dialects.arith", "mulf")
    mulf_op = mulf(c, c)

    assert isinstance(mulf_op, MulfOp)
    assert mulf_op.operands[0] == c.results[0]
    assert mulf_op.operands[1] == c.results[0]
    assert isinstance(mulf_op.results[0].type, Float32Type)


def test_resolves_overloads():
    a = ConstantOp.from_int_and_width(1, i64)
    b = ConstantOp.from_int_and_width(2, i64)

    addi = OpResolver.resolve_op_overload("__add__", builtin._Integer)
    addi_op = addi(a, b)

    assert isinstance(addi_op, AddiOp)
    assert addi_op.operands[0] == a.results[0]
    assert addi_op.operands[1] == b.results[0]
    assert isinstance(addi_op.results[0].type, IntegerType)
    assert addi_op.results[0].type.width.data == 64

    c = ConstantOp(FloatAttr(5.0, f64))
    mulf = OpResolver.resolve_op_overload("__mul__", builtin._Float64)
    mulf_op = mulf(c, c)

    assert isinstance(mulf_op, MulfOp)
    assert mulf_op.operands[0] == c.results[0]
    assert mulf_op.operands[1] == c.results[0]
    assert isinstance(mulf_op.results[0].type, Float64Type)
