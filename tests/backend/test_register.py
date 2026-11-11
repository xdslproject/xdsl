import re

import pytest

from xdsl.backend.register_type import RegisterType
from xdsl.context import Context
from xdsl.dialects.builtin import IntAttr, NoneAttr, StringAttr
from xdsl.dialects.test import Test, TestRegisterType
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import Parser
from xdsl.utils.exceptions import VerifyException


def test_register_clashes():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Infinite register prefix 'x' clashes with register names ['x0']."
        ),
    ):

        @irdl_attr_definition
        class ClashRegister(RegisterType):
            name = "test.reg_clash"

            @classmethod
            def index_by_name(cls) -> dict[str, int]:
                return {"x0": 0}

            @classmethod
            def infinite_register_prefix(cls):
                return "x"


def test_unallocated_register():
    assert not TestRegisterType.unallocated().is_allocated
    assert TestRegisterType.from_name("x0").is_allocated


def test_register_from_string():
    # Register with valid ABI name is fine
    assert TestRegisterType.from_name("x0").register_name == "a0"
    assert TestRegisterType.from_name("x0").index == IntAttr(0)
    assert TestRegisterType.from_name("x1").register_name == "a1"
    assert TestRegisterType.from_name("x1").index == IntAttr(1)
    assert TestRegisterType.from_name(StringAttr("x1")) == TestRegisterType.from_name(
        "a1"
    )
    assert (
        len({TestRegisterType.from_name("x0"), TestRegisterType.from_name("a0")}) == 1
    )

    # Register with infinite ABI name is fine
    assert TestRegisterType.from_name("y0").register_name == "y0"
    assert TestRegisterType.from_name("y0").index == IntAttr(-1)
    assert TestRegisterType.from_name("y1").register_name == "y1"
    assert TestRegisterType.from_name("y1").index == IntAttr(-2)


@pytest.mark.parametrize("name", ["foo", "yy", "y", "y01", "y+1", "y-1"])
def test_invalid_register_name(name: str):
    with pytest.raises(
        ValueError,
        match=re.escape(f"Invalid register name {name} for register type test.reg."),
    ):
        TestRegisterType.from_name(name)


@pytest.mark.parametrize(
    "index, name",
    [(NoneAttr(), ""), (IntAttr(0), "a0"), (IntAttr(-1), "y0"), (IntAttr(-4), "y3")],
)
def test_register_roundtrip(index: IntAttr | NoneAttr, name: str):
    register = TestRegisterType(index)
    assert register.parameters == (index,)
    assert register.register_name == name
    assert register.is_allocated == bool(name)
    assert register == TestRegisterType.from_name(name)
    assert register == TestRegisterType.new((index,))
    expected = f"!test.reg<{name}>" if name else "!test.reg"
    assert str(register) == expected
    ctx = Context()
    ctx.load_dialect(Test)
    assert Parser(ctx, expected).parse_type() == register


def test_invalid_index():
    with pytest.raises(
        ValueError, match=re.escape("Infinite index must be nonnegative, got -1.")
    ):
        TestRegisterType.infinite_register(-1)

    with pytest.raises(
        VerifyException, match=re.escape("Invalid index 2 for register type test.reg.")
    ):
        TestRegisterType(IntAttr(2))

    with pytest.raises(
        VerifyException, match=re.escape("Invalid index 2 for register type test.reg.")
    ):
        TestRegisterType.new((IntAttr(2),))


def test_name_by_index():
    assert TestRegisterType.abi_name_by_index() == {0: "a0", 1: "a1"}


def test_from_index():
    assert TestRegisterType.from_index(0) == TestRegisterType.from_name("a0")
    assert TestRegisterType.from_index(1) == TestRegisterType.from_name("a1")
    assert TestRegisterType.from_index(-1) == TestRegisterType.infinite_register(0)
    assert TestRegisterType.from_index(-2) == TestRegisterType.infinite_register(1)
