import pytest

from xdsl.dialects.builtin import (
    IntegerAttr,
    i32,
)
from xdsl.dialects.seq import (
    ClockDivider,
    clock,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import TestSSAValue


def test_clockdivider_verify():
    clock_div = ClockDivider(
        TestSSAValue(clock),
        IntegerAttr(512, i32),
    )
    with pytest.raises(
        VerifyException,
        match="Operation does not verify: pow2 has to be an 8-bit signless integer",
    ):
        clock_div.verify()
