import pytest
from xdsl.dialects.builtin import FloatAttr, f32
from xdsl.dialects.experimental.stencil import ReturnOp, ResultType, ApplyOp

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

    apply_op = ApplyOp.get([result_type_val1], [], f32, 2)

    assert len(apply_op.args) == 1
    assert len(apply_op.res) == 1
    assert apply_op.res[0].typ.element_type == f32
    assert len(apply_op.res[0].typ.shape) == 2


def test_stencil_apply_no_args():
    apply_op = ApplyOp.get([], [], f32, 1, result_count=2)

    assert len(apply_op.args) == 0
    assert len(apply_op.res) == 2
    assert apply_op.res[0].typ.element_type == f32
    assert len(apply_op.res[0].typ.shape) == 1


def test_stencil_apply_no_results():
    # Should error if there are no results expected
    with pytest.raises(AssertionError):
        apply_op = ApplyOp.get([], [], f32, 0)
