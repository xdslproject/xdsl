import pytest

from xdsl.dialects.builtin import IntegerType, i32
from xdsl.dialects.comb import ConcatOp, ICmpOp
from xdsl.dialects.test import TestOp, TestType
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import TestSSAValue


def test_icmp_incorrect_comparison():
    a = TestSSAValue(i32)
    b = TestSSAValue(i32)

    with pytest.raises(VerifyException) as e:
        # 'slet' is an invalid comparison operation
        _icmp_op = ICmpOp(a, b, "slet")
    assert e.value.args[0] == "Unknown comparison mnemonic: slet"


def test_comb_concat_builder():
    a = TestSSAValue(IntegerType(5))
    b = TestSSAValue(IntegerType(3))
    c = TestSSAValue(IntegerType(1))
    foo = TestSSAValue(TestType("foo"))

    concat = ConcatOp.from_int_values([a, b, c])
    assert concat is not None
    assert isinstance(concat.result.type, IntegerType)
    assert concat.result.type.width.data == 9

    bad_concat = ConcatOp.from_int_values([a, foo])
    assert bad_concat is None


def test_comb_concat_verifier():
    a = TestOp(result_types=[IntegerType(5)])
    b = TestOp(result_types=[IntegerType(3)])
    c = TestOp(result_types=[IntegerType(1)])

    ConcatOp([a, b, c], IntegerType(9)).verify()

    with pytest.raises(VerifyException):
        ConcatOp([a, b, c], IntegerType(2)).verify()
