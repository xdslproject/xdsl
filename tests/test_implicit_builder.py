from __future__ import annotations

import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, test


@pytest.fixture
def dialect_context() -> Context:
    """Fixture that creates a Context with Arith and Builtin dialects loaded.

    Returns:
        Context: An MLIR Context with required dialects.
    """
    ctx = Context()
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(builtin.Builtin)
    return ctx


def test_no_implicit_builder(dialect_context: Context) -> None:
    """Test creating an operation without an implict builder."""
    assert dialect_context
    expected = 'builtin.module {\n  %i = "test.op"() : () -> i32\n}'

    module = builtin.ModuleOp([])
    i = test.TestOp.create(result_types=[builtin.i32])
    i.results[0].name_hint = "i"
    module.body.block.add_op(i)
    assert str(module) == expected


def test_single_implicit_builder(dialect_context: Context):
    """Test creating an operation with a single implict builder."""
    assert dialect_context
    expected = 'builtin.module {\n  %i = "test.op"() : () -> i32\n}'

    with ImplicitBuilder((module := builtin.ModuleOp([])).body):
        assert len(ImplicitBuilder._stack.stack) == 1  # pyright: ignore[reportPrivateUsage]
        i = test.TestOp.create(result_types=[builtin.i32])
        i.results[0].name_hint = "i"
    assert str(module) == expected


def test_nested_implicit_builders(dialect_context: Context):
    """Test creating an operation with a single implict builder."""
    assert dialect_context
    expected = (
        'builtin.module {\n  builtin.module {\n    %i = "test.op"() : () -> i32\n  }\n}'
    )

    with ImplicitBuilder((module_outer := builtin.ModuleOp([])).body):
        assert len(ImplicitBuilder._stack.stack) == 1  # pyright: ignore[reportPrivateUsage]
        with ImplicitBuilder((_module_inner := builtin.ModuleOp([])).body):
            assert len(ImplicitBuilder._stack.stack) == 2  # pyright: ignore[reportPrivateUsage]
            i = test.TestOp.create(result_types=[builtin.i32])
            i.results[0].name_hint = "i"
    assert str(module_outer) == expected
