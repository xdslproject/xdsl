import pytest

from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    i1,
    i32,
)
from xdsl.dialects.seq import (
    ClockDividerOp,
    CompRegOp,
    clock,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import TestSSAValue


def test_clockdivider_verify():
    clock_div = ClockDividerOp(
        TestSSAValue(clock),
        IntegerAttr(512, i32),
    )
    with pytest.raises(
        VerifyException,
        match="Operation does not verify: pow2 has to be an 8-bit signless integer",
    ):
        clock_div.verify()


def test_compreg_builder():
    data_val = TestSSAValue(IntegerType(5))
    bool_val = TestSSAValue(i1)
    clock_val = TestSSAValue(clock)

    CompRegOp(data_val, clock_val).verify()

    with_reset = CompRegOp(data_val, clock_val, (bool_val, data_val))
    with_reset.verify()
    assert with_reset.reset is not None
