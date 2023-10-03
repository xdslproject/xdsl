import pytest

from xdsl.dialects.builtin import i32
from xdsl.dialects.comb import ICmpOp
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import TestSSAValue


def test_icmp_incorrect_comparison():
    a = TestSSAValue(i32)
    b = TestSSAValue(i32)

    with pytest.raises(VerifyException) as e:
        # 'slet' is an invalid comparison operation
        _icmp_op = ICmpOp(a, b, "slet")
    assert e.value.args[0] == "Unknown comparison mnemonic: slet"
