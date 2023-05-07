from xdsl.utils.test_value import TestSSAValue
from xdsl.dialects import riscv, snitch

from xdsl.dialects.builtin import IntegerAttr, i32

from xdsl.utils.exceptions import VerifyException

import pytest


def test_csr_op():
    stream = TestSSAValue(riscv.RegisterType(riscv.Registers.A1))
    value = TestSSAValue(riscv.RegisterType(riscv.Registers.A1))
    valid = IntegerAttr(snitch.SnitchResources.dimensions - 1, i32)
    invalid = IntegerAttr(snitch.SnitchResources.dimensions, i32)

    snitch.SsrSetDimensionBoundOp(stream=stream, value=value, dimension=valid).verify()
    with pytest.raises(VerifyException):
        snitch.SsrSetDimensionBoundOp(
            stream=stream, value=value, dimension=invalid
        ).verify()
    snitch.SsrSetDimensionStrideOp(stream=stream, value=value, dimension=valid).verify()
    with pytest.raises(VerifyException):
        snitch.SsrSetDimensionStrideOp(
            stream=stream, value=value, dimension=invalid
        ).verify()
    snitch.SsrSetDimensionSourceOp(stream=stream, value=value, dimension=valid).verify()
    with pytest.raises(VerifyException):
        snitch.SsrSetDimensionSourceOp(
            stream=stream, value=value, dimension=invalid
        ).verify()
    snitch.SsrSetDimensionDestinationOp(
        stream=stream, value=value, dimension=valid
    ).verify()
    with pytest.raises(VerifyException):
        snitch.SsrSetDimensionDestinationOp(
            stream=stream, value=value, dimension=invalid
        ).verify()
