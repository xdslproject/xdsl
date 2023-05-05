import pytest
from xdsl.dialects.builtin import FloatAttr, f32
from xdsl.dialects.experimental.stencil import (
    ReturnOp,
    ResultType,
    ApplyOp,
    TempType,
    LoadOp,
    FieldType,
    IndexAttr,
)
from xdsl.ir import Block
from xdsl.utils.test_value import TestSSAValue


def test_stencil_return_single_float():
    float_val1 = TestSSAValue(FloatAttr(4.0, f32))
    return_op = ReturnOp.get([float_val1])

    assert return_op.arg[0] is float_val1


def test_stencil_return_multiple_floats():
    float_val1 = TestSSAValue(FloatAttr(4.0, f32))
    float_val2 = TestSSAValue(FloatAttr(5.0, f32))
    float_val3 = TestSSAValue(FloatAttr(6.0, f32))

    return_op = ReturnOp.get([float_val1, float_val2, float_val3])

    assert return_op.arg[0] is float_val1
    assert return_op.arg[1] is float_val2
    assert return_op.arg[2] is float_val3


def test_stencil_return_single_ResultType():
    result_type_val1 = TestSSAValue(ResultType.from_type(f32))
    return_op = ReturnOp.get([result_type_val1])

    assert return_op.arg[0] is result_type_val1


def test_stencil_return_multiple_ResultType():
    result_type_val1 = TestSSAValue(ResultType.from_type(f32))
    result_type_val2 = TestSSAValue(ResultType.from_type(f32))
    result_type_val3 = TestSSAValue(ResultType.from_type(f32))

    return_op = ReturnOp.get([result_type_val1, result_type_val2, result_type_val3])

    assert return_op.arg[0] is result_type_val1
    assert return_op.arg[1] is result_type_val2
    assert return_op.arg[2] is result_type_val3


def test_stencil_apply():
    result_type_val1 = TestSSAValue(ResultType.from_type(f32))

    stencil_temptype = TempType.from_shape([-1] * 2, f32)
    apply_op = ApplyOp.get([result_type_val1], Block([]), [stencil_temptype])

    assert len(apply_op.args) == 1
    assert len(apply_op.res) == 1
    assert isinstance(apply_op.res[0].typ, TempType)
    assert len(apply_op.res[0].typ.shape) == 2


def test_stencil_apply_no_args():
    stencil_temptype = TempType.from_shape([-1] * 1, f32)
    apply_op = ApplyOp.get([], Block([]), [stencil_temptype, stencil_temptype])

    assert len(apply_op.args) == 0
    assert len(apply_op.res) == 2
    assert isinstance(apply_op.res[0].typ, TempType)
    assert len(apply_op.res[0].typ.shape) == 1


def test_stencil_apply_no_results():
    # Should error if there are no results expected
    with pytest.raises(AssertionError):
        ApplyOp.get([], Block([]), [])


def test_stencil_load():
    field_type = FieldType.from_shape([1, 1], f32)
    result_type_val1 = TestSSAValue(field_type)

    load = LoadOp.get(result_type_val1)

    assert isinstance(load.field.typ, FieldType)
    assert load.field.typ == field_type
    assert len(load.field.typ.shape) == 2
    assert load.lb is None
    assert load.ub is None


def test_stencil_load_bounds():
    field_type = FieldType.from_shape([1, 1], f32)
    result_type_val1 = TestSSAValue(field_type)

    lb = IndexAttr.get(1, 1)
    ub = IndexAttr.get(64, 64)

    load = LoadOp.get(result_type_val1, lb, ub)

    assert isinstance(load.lb, IndexAttr)
    assert len(load.lb.array) == 2
    for my_val, load_val in zip(lb.array.data, load.lb.array):
        assert my_val.value.data == load_val.value.data
    assert isinstance(load.ub, IndexAttr)
    assert len(load.ub.array) == 2
    for my_val, load_val in zip(ub.array.data, load.ub.array):
        assert my_val.value.data == load_val.value.data
