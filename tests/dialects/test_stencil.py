import pytest

from xdsl.dialects.stencil import CastOp
from xdsl.utils.exceptions import VerifyException
from xdsl.dialects.builtin import FloatAttr, f32, f64
from xdsl.dialects.experimental.stencil import (
    ReturnOp,
    ResultType,
    FieldType,
    IndexAttr,
)

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


def test_stencil_cast_op_verifier():
    field = TestSSAValue(FieldType.from_shape((-1, -1, -1), f32))

    # check that correct op verifies correctly
    cast = CastOp.get(
        field,
        IndexAttr.get(-2, -2, -2),
        IndexAttr.get(100, 100, 100),
        FieldType.from_shape((102, 102, 102), f32),
    )
    cast.verify()

    # check that math is correct
    with pytest.raises(VerifyException) as ex1:
        cast = CastOp.get(
            field,
            IndexAttr.get(-2, -2, -2),
            IndexAttr.get(100, 100, 100),
            FieldType.from_shape((100, 100, 100), f32),
        )
        cast.verify()
    assert "math" in ex1.value.args[0]

    # check that output has same dims as input and lb, ub
    with pytest.raises(VerifyException) as ex2:
        cast = CastOp.get(
            field,
            IndexAttr.get(-2, -2, -2),
            IndexAttr.get(100, 100, 100),
            FieldType.from_shape((102, 102), f32),
        )
        cast.verify()
    assert "same dimensions" in ex2.value.args[0]

    # check that input has same shape as lb, ub, output
    with pytest.raises(VerifyException) as ex3:
        dyn_field_wrong_shape = TestSSAValue(FieldType.from_shape((-1, -1), f32))
        cast = CastOp.get(
            dyn_field_wrong_shape,
            IndexAttr.get(-2, -2, -2),
            IndexAttr.get(100, 100, 100),
            FieldType.from_shape((102, 102, 102), f32),
        )
        cast.verify()
    assert "same dimensions" in ex3.value.args[0]

    # check that input and output have same element type
    with pytest.raises(VerifyException) as ex4:
        cast = CastOp.get(
            field,
            IndexAttr.get(-2, -2, -2),
            IndexAttr.get(100, 100, 100),
            FieldType.from_shape((102, 102, 102), f64),
        )
        cast.verify()
    assert "element type" in ex4.value.args[0]

    # check that len(lb) == len(ub)
    with pytest.raises(VerifyException) as ex5:
        cast = CastOp.get(
            field,
            IndexAttr.get(
                -2,
                -2,
            ),
            IndexAttr.get(100, 100, 100),
            FieldType.from_shape((102, 102, 102), f32),
        )
        cast.verify()
    assert "same dimensions" in ex5.value.args[0]

    # check that len(lb) == len(ub)
    with pytest.raises(VerifyException) as ex6:
        cast = CastOp.get(
            field,
            IndexAttr.get(-2, -2, -2),
            IndexAttr.get(100, 100),
            FieldType.from_shape((102, 102, 102), f32),
        )
        cast.verify()
    assert "same dimensions" in ex6.value.args[0]

    # check that input must be dynamic
    with pytest.raises(VerifyException) as ex7:
        non_dyn_field = TestSSAValue(FieldType.from_shape((102, 102, 102), f32))
        cast = CastOp.get(
            non_dyn_field,
            IndexAttr.get(-2, -2, -2),
            IndexAttr.get(100, 100, 100),
            FieldType.from_shape((102, 102, 102), f32),
        )
        cast.verify()
    assert "dynamic" in ex7.value.args[0]


def test_cast_op_constructor():
    field = TestSSAValue(FieldType.from_shape((-1, -1, -1), f32))

    cast = CastOp.get(
        field,
        IndexAttr.get(-2, -3, -4),
        IndexAttr.get(100, 100, 0),
    )

    assert cast.result.typ == FieldType.from_shape((102, 103, 4), f32)
