import pytest

from xdsl.dialects.arm.assembly import assembly_arg_str
from xdsl.dialects.arm.register import IntRegisterType
from xdsl.dialects.test import TestType
from xdsl.utils.test_value import TestSSAValue


def test_assembly_arg_str_ARMRegister():
    arg = IntRegisterType("x0")
    assert assembly_arg_str(arg) == "x0"


def test_assembly_arg_str_SSAValue_valid():
    arg = TestSSAValue(IntRegisterType("x1"))
    assert assembly_arg_str(arg) == "x1"


def test_assembly_arg_str_SSAValue_invalid():
    arg = TestSSAValue(TestType("foo"))
    with pytest.raises(ValueError, match="Unexpected argument type"):
        assembly_arg_str(arg)
