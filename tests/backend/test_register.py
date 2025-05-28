import pytest

from xdsl.backend.register_type import RegisterType
from xdsl.dialects.builtin import IntAttr, NoneAttr, StringAttr
from xdsl.irdl import irdl_attr_definition
from xdsl.utils.exceptions import VerifyException


def test_register_clashes():
    @irdl_attr_definition
    class ClashRegister(RegisterType):
        name = "test.reg_clash"

        def verify(self) -> None: ...

        @classmethod
        def index_by_name(cls) -> dict[str, int]:
            return {"x0": 0}

        @classmethod
        def infinite_register_prefix(cls):
            return "x"

    with pytest.raises(
        AssertionError,
        match="Invalid 'infinite' register name: x0 clashes with finite register set",
    ):
        ClashRegister.infinite_register(0)


@irdl_attr_definition
class TestRegister(RegisterType):
    name = "test.reg"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return {"x0": 0, "x1": 1, "a0": 0, "a1": 1}

    @classmethod
    def infinite_register_prefix(cls):
        return "y"


def test_unallocated_register():
    assert not TestRegister.unallocated().is_allocated
    assert TestRegister.from_name("x0").is_allocated


def test_register_from_string():
    # Register with valid ABI name is fine
    assert TestRegister.from_name("x0").register_name == StringAttr("x0")
    assert TestRegister.from_name("x0").index == IntAttr(0)
    assert TestRegister.from_name("x1").register_name == StringAttr("x1")
    assert TestRegister.from_name("x1").index == IntAttr(1)

    # Register with infinite ABI name is fine
    assert TestRegister.from_name("y0").register_name == StringAttr("y0")
    assert TestRegister.from_name("y0").index == IntAttr(-1)
    assert TestRegister.from_name("y1").register_name == StringAttr("y1")
    assert TestRegister.from_name("y1").index == IntAttr(-2)

    # Infinite prefix but not a number
    with pytest.raises(
        VerifyException,
        match="Invalid register name yy for register type test.reg",
    ):
        TestRegister.from_name("yy")

    # Incorrect name
    with pytest.raises(
        VerifyException, match="Invalid register name z0 for register type test.reg"
    ):
        TestRegister.from_name("z0")


def test_invalid_register_name():
    with pytest.raises(
        VerifyException, match="Invalid register name foo for register type test.reg."
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
        VerifyException, match="Missing index for register x1, expected 1."
    ):
        TestRegister(NoneAttr(), StringAttr("x1"))

    with pytest.raises(
        VerifyException, match="Invalid index 2 for register x1, expected 1."
    ):
        TestRegister(IntAttr(2), StringAttr("x1"))


def test_name_by_index():
    assert TestRegister.abi_name_by_index() == {0: "a0", 1: "a1"}


def test_from_index():
    assert TestRegister.from_index(0) == TestRegister.from_name("a0")
    assert TestRegister.from_index(1) == TestRegister.from_name("a1")
    assert TestRegister.from_index(-1) == TestRegister.infinite_register(0)
    assert TestRegister.from_index(-2) == TestRegister.infinite_register(1)
