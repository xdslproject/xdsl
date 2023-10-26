import pytest

from xdsl.dialects import riscv, snitch
from xdsl.dialects.builtin import IntAttr
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import TestSSAValue


def test_csr_op():
    stream = TestSSAValue(riscv.Registers.A1)
    value = TestSSAValue(riscv.Registers.A1)
    valid = IntAttr(snitch.SnitchResources.dimensions - 1)
    invalid = IntAttr(snitch.SnitchResources.dimensions)

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
