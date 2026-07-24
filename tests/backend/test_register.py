import re

import pytest

from xdsl.backend.register_type import RegisterType
from xdsl.dialects.builtin import IntAttr, NoneAttr, StringAttr
from xdsl.dialects.test import TestRegisterType
from xdsl.irdl import irdl_attr_definition
from xdsl.utils.exceptions import VerifyException


def test_register_clashes():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Infinite register prefix 'x' clashes with register names ['x0']."
        ),
    ):

        @irdl_attr_definition
        class ClashRegister(RegisterType):  # pyright: ignore[reportUnusedClass]
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
    assert TestRegisterType.from_name("x0").register_name == StringAttr("x0")
    assert TestRegisterType.from_name("x0").index == IntAttr(0)
    assert TestRegisterType.from_name("x1").register_name == StringAttr("x1")
    assert TestRegisterType.from_name("x1").index == IntAttr(1)

    # Register with infinite ABI name is fine
    assert TestRegisterType.from_name("y0").register_name == StringAttr("y0")
    assert TestRegisterType.from_name("y0").index == IntAttr(-1)
    assert TestRegisterType.from_name("y1").register_name == StringAttr("y1")
    assert TestRegisterType.from_name("y1").index == IntAttr(-2)

    # Infinite prefix but not a number
    with pytest.raises(
        VerifyException,
        match="Invalid register name yy for register type test.reg",
    ):
        TestRegisterType.from_name("yy")

    # Incorrect name
    with pytest.raises(
        VerifyException, match="Invalid register name z0 for register type test.reg"
    ):
        TestRegisterType.from_name("z0")


def test_invalid_register_name():
    with pytest.raises(
        VerifyException, match="Invalid register name foo for register type test.reg."
    ):
        TestRegisterType.from_name("foo")


def test_invalid_index():
    with pytest.raises(
        AssertionError, match="Infinite index must be positive, got -1."
    ):
        TestRegisterType.infinite_register(-1)

    with pytest.raises(
        VerifyException, match="Invalid index 1 for unallocated register."
    ):
        TestRegisterType(IntAttr(1), StringAttr(""))

    with pytest.raises(
        VerifyException, match="Missing index for register x1, expected 1."
    ):
        TestRegisterType(NoneAttr(), StringAttr("x1"))

    with pytest.raises(
        VerifyException, match="Invalid index 2 for register x1, expected 1."
    ):
        TestRegisterType(IntAttr(2), StringAttr("x1"))


def test_name_by_index():
    assert TestRegisterType.abi_name_by_index() == {0: "a0", 1: "a1"}


def test_from_index():
    assert TestRegisterType.from_index(0) == TestRegisterType.from_name("a0")
    assert TestRegisterType.from_index(1) == TestRegisterType.from_name("a1")
    assert TestRegisterType.from_index(-1) == TestRegisterType.infinite_register(0)
    assert TestRegisterType.from_index(-2) == TestRegisterType.infinite_register(1)
