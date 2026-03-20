import re

import pytest
from typing_extensions import override

from xdsl.backend.register_type import NamedRegisterType
from xdsl.dialects.builtin import IntAttr, StringAttr
from xdsl.irdl import irdl_attr_definition


def test_register_clashes():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Infinite register prefix 'x' clashes with register names ['x0']."
        ),
    ):

        @irdl_attr_definition
        class ClashRegister(NamedRegisterType):  # pyright: ignore[reportUnusedClass]
            name = "test.reg_clash"

            @override
            @classmethod
            def index_by_name(cls) -> dict[str, int]:
                return {"x0": 0}

            @override
            @classmethod
            def infinite_register_prefix(cls):
                return "x"


@irdl_attr_definition
class TestRegister(NamedRegisterType):
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
    assert TestRegister.from_name("x0").register_name == StringAttr("a0")
    assert TestRegister.from_name("x0").index == IntAttr(0)
    assert TestRegister.from_name("x1").register_name == StringAttr("a1")
    assert TestRegister.from_name("x1").index == IntAttr(1)

    # Register with infinite ABI name is fine
    assert TestRegister.from_name("y0").register_name == StringAttr("y0")
    assert TestRegister.from_name("y0").index == IntAttr(-1)
    assert TestRegister.from_name("y1").register_name == StringAttr("y1")
    assert TestRegister.from_name("y1").index == IntAttr(-2)

    # Infinite prefix but not a number
    with pytest.raises(
        ValueError,
        match="Invalid register name yy for register type test.reg",
    ):
        TestRegister.from_name("yy")

    # Incorrect name
    with pytest.raises(
        ValueError, match="Invalid register name z0 for register type test.reg"
    ):
        TestRegister.from_name("z0")


def test_invalid_register_name():
    with pytest.raises(
        ValueError, match="Invalid register name foo for register type test.reg."
    ):
        TestRegister.from_name("foo")


def test_invalid_index():
    with pytest.raises(
        AssertionError, match="Infinite index must be positive, got -1."
    ):
        TestRegister.infinite_register(-1)


def test_name_by_index():
    assert TestRegister.abi_name_by_index() == {0: "a0", 1: "a1"}


def test_from_index():
    assert TestRegister.from_index(0) == TestRegister.from_name("a0")
    assert TestRegister.from_index(1) == TestRegister.from_name("a1")
    assert TestRegister.from_index(-1) == TestRegister.infinite_register(0)
    assert TestRegister.from_index(-2) == TestRegister.infinite_register(1)
