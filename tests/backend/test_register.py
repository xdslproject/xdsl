import pytest

from xdsl.backend.register_type import RegisterType
from xdsl.dialects.builtin import IntAttr, NoneAttr, StringAttr
from xdsl.irdl import irdl_attr_definition
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class TestRegister(RegisterType):
    name = "test.reg"

    @classmethod
    def instruction_set_name(cls) -> str:
        return "TEST"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return {"x0": 0, "y0": 1}

    @classmethod
    def infinite_register_prefix(cls):
        return "x"


def test_register_clashes():
    with pytest.raises(
        AssertionError,
        match="Invalid 'infinite' register name: x0 clashes with finite register set",
    ):
        TestRegister.infinite_register(0)


def test_unallocated_register():
    assert not TestRegister.unallocated().is_allocated
    assert TestRegister.from_name("x0").is_allocated


def test_invalid_register_name():
    with pytest.raises(
        VerifyException, match="Invalid register name foo for register set TEST."
    ):
        TestRegister.from_name("foo")


def test_invalid_index():
    with pytest.raises(
        AssertionError, match="Infinite index must be positive, got -1."
    ):
        TestRegister.infinite_register(-1)

    with pytest.raises(
        VerifyException, match="Invalid index 1 for unallocated register."
    ):
        TestRegister(IntAttr(1), StringAttr(""))

    with pytest.raises(
        VerifyException, match="Missing index for register y0, expected 1."
    ):
        TestRegister(NoneAttr(), StringAttr("y0"))

    with pytest.raises(
        VerifyException, match="Invalid index for register y0 2, expected 1."
    ):
        TestRegister(IntAttr(2), StringAttr("y0"))
