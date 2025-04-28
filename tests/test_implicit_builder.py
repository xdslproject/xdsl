from __future__ import annotations

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import builtin, test


def test_no_implicit_builder() -> None:
    """Test creating an operation without an implict builder."""
    expected = """\
builtin.module {
  %0 = "test.op"() : () -> i32
}"""

    module = builtin.ModuleOp([])
    i = test.TestOp.create(result_types=[builtin.i32])
    module.body.block.add_op(i)
    assert str(module) == expected


def test_single_implicit_builder() -> None:
    """Test creating an operation with a single implict builder."""
    expected = """\
builtin.module {
  %0 = "test.op"() : () -> i32
}"""

    with ImplicitBuilder((module := builtin.ModuleOp([])).body):
        assert len(ImplicitBuilder._stack.stack) == 1  # pyright: ignore[reportPrivateUsage]
        _i = test.TestOp.create(result_types=[builtin.i32])
    assert str(module) == expected


def test_nested_implicit_builders() -> None:
    """Test creating an operation with a single nested implict builder."""
    expected = """\
builtin.module {
  builtin.module {
    %0 = "test.op"() : () -> i32
  }
}"""

    with ImplicitBuilder((module_outer := builtin.ModuleOp([])).body):
        assert len(ImplicitBuilder._stack.stack) == 1  # pyright: ignore[reportPrivateUsage]
        with ImplicitBuilder((_module_inner := builtin.ModuleOp([])).body):
            assert len(ImplicitBuilder._stack.stack) == 2  # pyright: ignore[reportPrivateUsage]
            _i = test.TestOp.create(result_types=[builtin.i32])
    assert str(module_outer) == expected
