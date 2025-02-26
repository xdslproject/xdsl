import pytest

from xdsl.backend.register_type import RegisterType
from xdsl.dialects.builtin import IntAttr, StringAttr
from xdsl.irdl import irdl_attr_definition


def test_register_clashes():
    @irdl_attr_definition
    class ClashRegister(RegisterType):
        name = "test.reg_clash"

        def verify(self) -> None: ...

        @classmethod
        def instruction_set_name(cls) -> str:
            return "TEST"

        @classmethod
        def abi_index_by_name(cls) -> dict[str, int]:
            return {"x0": 0}

        @classmethod
        def infinite_register_prefix(cls):
            return "x"

    with pytest.raises(
        AssertionError,
        match="Invalid 'infinite' register name: x0 clashes with finite register set",
    ):
        ClashRegister.infinite_register(0)


def test_register_from_string():
    @irdl_attr_definition
    class TestRegister(RegisterType):
        name = "test.reg"

        def verify(self) -> None: ...

        @classmethod
        def instruction_set_name(cls) -> str:
            return "TEST"

        @classmethod
        def abi_index_by_name(cls) -> dict[str, int]:
            return {"x0": 0, "x1": 1}

        @classmethod
        def infinite_register_prefix(cls):
            return "y"

    # Register with valid ABI name is fine
    assert TestRegister("x0").spelling == StringAttr("x0")
    assert TestRegister("x0").index == IntAttr(0)
    assert TestRegister("x1").spelling == StringAttr("x1")
    assert TestRegister("x1").index == IntAttr(1)

    # Register with infinite ABI name is fine
    assert TestRegister("y0").spelling == StringAttr("y0")
    assert TestRegister("y0").index == IntAttr(-1)
    assert TestRegister("y1").spelling == StringAttr("y1")
    assert TestRegister("y1").index == IntAttr(-2)

    # Infinite prefix but not a number
    with pytest.raises(
        ValueError, match="Invalid register spelling yy for class TestRegister"
    ):
        TestRegister("yy")

    # Incorrect name
    with pytest.raises(
        ValueError, match="Invalid register spelling z0 for class TestRegister"
    ):
        TestRegister("z0")
