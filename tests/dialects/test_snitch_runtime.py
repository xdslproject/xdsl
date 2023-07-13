import pytest

from xdsl.dialects import arith, builtin, snitch_runtime
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import TestSSAValue


def test_ssr_loop_op():
    data_mover = TestSSAValue(builtin.i32)
    ten = arith.Constant.from_int_and_width(10, builtin.IndexType())
    bounds1d = [ten]
    bounds2d = [ten, ten]
    bounds3d = [ten, ten, ten]
    bounds4d = [ten, ten, ten, ten]
    strides1d = [ten]
    strides2d = [ten, ten]
    strides3d = [ten, ten, ten]
    strides4d = [ten, ten, ten, ten]

    # checking for valid and invalid var_operand lengths for the individual ops
    snitch_runtime.SsrLoop1dOp(
        data_mover=data_mover, bounds=bounds1d, strides=strides1d
    )
    with pytest.raises(VerifyException):
        snitch_runtime.SsrLoop1dOp(
            data_mover=data_mover, bounds=bounds2d, strides=strides2d
        ).verify()
    snitch_runtime.SsrLoop2dOp(
        data_mover=data_mover, bounds=bounds2d, strides=strides2d
    )
    with pytest.raises(VerifyException):
        snitch_runtime.SsrLoop2dOp(
            data_mover=data_mover, bounds=bounds3d, strides=strides3d
        ).verify()
    snitch_runtime.SsrLoop3dOp(
        data_mover=data_mover, bounds=bounds3d, strides=strides3d
    )
    with pytest.raises(VerifyException):
        snitch_runtime.SsrLoop3dOp(
            data_mover=data_mover, bounds=bounds4d, strides=strides4d
        ).verify()
    snitch_runtime.SsrLoop4dOp(
        data_mover=data_mover, bounds=bounds4d, strides=strides4d
    )
    with pytest.raises(VerifyException):
        snitch_runtime.SsrLoop4dOp(
            data_mover=data_mover, bounds=bounds1d, strides=strides1d
        ).verify()

    # checking for invalid combinations of var_operand lengths
    with pytest.raises(VerifyException):
        snitch_runtime.SsrLoop1dOp(
            data_mover=data_mover, bounds=bounds1d, strides=strides3d
        ).verify()
    with pytest.raises(VerifyException):
        snitch_runtime.SsrLoop2dOp(
            data_mover=data_mover, bounds=bounds4d, strides=strides1d
        ).verify()
    with pytest.raises(VerifyException):
        snitch_runtime.SsrLoop3dOp(
            data_mover=data_mover, bounds=bounds2d, strides=strides3d
        ).verify()
    with pytest.raises(VerifyException):
        snitch_runtime.SsrLoop4dOp(
            data_mover=data_mover, bounds=bounds3d, strides=strides2d
        ).verify()
