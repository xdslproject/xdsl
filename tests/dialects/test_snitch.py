import pytest

from xdsl.dialects import riscv, snitch
from xdsl.dialects.builtin import IntAttr
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


def test_csr_op():
    value = create_ssa_value(riscv.Registers.A1)
    stream = IntAttr(0)
    valid = IntAttr(snitch.SnitchResources.dimensions - 1)
    invalid = IntAttr(snitch.SnitchResources.dimensions)

    snitch.SsrSetDimensionBoundOp(value=value, dm=stream, dimension=valid).verify()
    with pytest.raises(VerifyException):
        snitch.SsrSetDimensionBoundOp(
            value=value, dm=stream, dimension=invalid
        ).verify()
    snitch.SsrSetDimensionStrideOp(value=value, dm=stream, dimension=valid).verify()
    with pytest.raises(VerifyException):
        snitch.SsrSetDimensionStrideOp(
            value=value, dm=stream, dimension=invalid
        ).verify()
    snitch.SsrSetDimensionSourceOp(value=value, dm=stream, dimension=valid).verify()
    with pytest.raises(VerifyException):
        snitch.SsrSetDimensionSourceOp(
            value=value, dm=stream, dimension=invalid
        ).verify()
    snitch.SsrSetDimensionDestinationOp(
        value=value, dm=stream, dimension=valid
    ).verify()
    with pytest.raises(VerifyException):
        snitch.SsrSetDimensionDestinationOp(
            value=value, dm=stream, dimension=invalid
        ).verify()
