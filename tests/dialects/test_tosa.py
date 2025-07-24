import pytest

from xdsl.dialects.builtin import (
    TensorType,
    i32,
    f32,
)
from xdsl.dialects.tosa import AddOp
from xdsl.utils.test_value import create_ssa_value
from xdsl.utils.exceptions import VerifyException


tensor_i = create_ssa_value(TensorType(i32, [1, 2, 3, 4]))
tensor_flat = create_ssa_value(TensorType(i32, [1, 1, 1, 1]))
tensor_f = create_ssa_value(TensorType(f32, [1, 2, 3, 4]))


def test_valid_cases():
    AddOp(operands=[tensor_i, tensor_i], result_types=[tensor_i.type]).verify_()
    AddOp(operands=[tensor_i, tensor_flat], result_types=[tensor_i.type]).verify_()


def test_invalid_cases():
    with pytest.raises(VerifyException):
        AddOp(operands=[tensor_i, tensor_f], result_types=[tensor_f.type]).verify_()

    with pytest.raises(VerifyException):
        AddOp(
            operands=[tensor_i, tensor_flat], result_types=[tensor_flat.type]
        ).verify_()
