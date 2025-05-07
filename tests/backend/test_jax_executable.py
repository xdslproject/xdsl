import re

import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import func, stablehlo
from xdsl.dialects.builtin import ModuleOp, StringAttr, TensorType, i32
from xdsl.irdl import IRDLOperation, attr_def, irdl_op_definition, traits_def
from xdsl.traits import SymbolOpInterface

pytest.importorskip("jax")

import jax  # noqa: E402

from xdsl.backend.jax_executable import JaxExecutable, array  # noqa: E402


def test_abs():
    TI32 = TensorType(i32, ())

    main_op = func.FuncOp("main", ((TI32,), (TI32,)))
    with ImplicitBuilder(main_op.body) as (arg,):
        res = stablehlo.AbsOp(arg).result
        func.ReturnOp(res)

    module = ModuleOp([main_op])

    executable = JaxExecutable.compile(module)

    assert executable.execute([array(-2, dtype=jax.numpy.int32)]) == [
        array(2, dtype=jax.numpy.int32)
    ]
    assert executable.execute([array(0, dtype=jax.numpy.int32)]) == [
        array(0, dtype=jax.numpy.int32)
    ]
    assert executable.execute([array(2, dtype=jax.numpy.int32)]) == [
        array(2, dtype=jax.numpy.int32)
    ]

    @executable
    def abs_tuple(a: jax.Array) -> tuple[jax.Array]: ...

    assert abs_tuple(array(-2, dtype=jax.numpy.int32)) == (
        array(2, dtype=jax.numpy.int32),
    )
    assert abs_tuple(array(0, dtype=jax.numpy.int32)) == (
        array(0, dtype=jax.numpy.int32),
    )
    assert abs_tuple(array(2, dtype=jax.numpy.int32)) == (
        array(2, dtype=jax.numpy.int32),
    )

    @executable
    def abs_one(a: jax.Array) -> jax.Array: ...

    assert abs_one(array(-2, dtype=jax.numpy.int32)) == array(2, dtype=jax.numpy.int32)
    assert abs_one(array(0, dtype=jax.numpy.int32)) == array(0, dtype=jax.numpy.int32)
    assert abs_one(array(2, dtype=jax.numpy.int32)) == array(2, dtype=jax.numpy.int32)


def test_add_sub():
    TI32 = TensorType(i32, ())

    main_op = func.FuncOp("main", ((TI32, TI32), (TI32, TI32)))
    with ImplicitBuilder(main_op.body) as (arg0, arg1):
        res0 = stablehlo.AddOp(arg0, arg1).result
        res1 = stablehlo.SubtractOp(arg0, arg1).result
        func.ReturnOp(res0, res1)

    module = ModuleOp([main_op])

    executable = JaxExecutable.compile(module)

    def a(i: int) -> jax.Array:
        return array(i, dtype=jax.numpy.int32)

    assert executable.execute([a(-2), a(-3)]) == [a(-5), a(1)]

    @executable
    def add_sub_tuple(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]: ...

    assert add_sub_tuple(a(-2), a(-3)) == (a(-5), a(1))


def test_no_main():
    with pytest.raises(ValueError, match="No `main` function in module"):
        module = ModuleOp([])
        JaxExecutable.compile(module)

    TI32 = TensorType(i32, ())

    with pytest.raises(ValueError, match="No `main` function in module"):
        main_op = func.FuncOp("not_main", ((TI32,), (TI32,)))
        with ImplicitBuilder(main_op.body) as (arg,):
            res = stablehlo.AbsOp(arg).result
            func.ReturnOp(res)

        module = ModuleOp([main_op])

        JaxExecutable.compile(module)


def test_main_not_func():
    @irdl_op_definition
    class SymNameOp(IRDLOperation):
        name = "sym_name"

        sym_name = attr_def(StringAttr)
        traits = traits_def(SymbolOpInterface())

    module = ModuleOp([SymNameOp(attributes={"sym_name": StringAttr("main")})])

    with pytest.raises(ValueError, match="`main` operation is not a `func.func`"):
        JaxExecutable.compile(module)


def test_parameter_count_mismatch():
    TI32 = TensorType(i32, ())

    main_op = func.FuncOp("main", ((TI32,), (TI32,)))
    with ImplicitBuilder(main_op.body) as (arg,):
        res = stablehlo.AbsOp(arg).result
        func.ReturnOp(res)

    module = ModuleOp([main_op])
    executable = JaxExecutable.compile(module)

    with pytest.raises(
        ValueError,
        match="Number of parameters .* does not match the number of operand types",
    ):

        @executable
        def abs_two_params(a: jax.Array, b: jax.Array) -> jax.Array: ...  # pyright: ignore[reportUnusedFunction]


def test_parameter_annotation():
    TI32 = TensorType(i32, ())

    main_op = func.FuncOp("main", ((TI32,), (TI32,)))
    with ImplicitBuilder(main_op.body) as (arg,):
        res = stablehlo.AbsOp(arg).result
        func.ReturnOp(res)

    module = ModuleOp([main_op])
    executable = JaxExecutable.compile(module)

    with pytest.raises(
        NotImplementedError, match="Parameter .* is not annotated as jnp.ndarray"
    ):

        @executable
        def abs_wrong_annotation(a: int) -> jax.Array: ...  # pyright: ignore[reportUnusedFunction]


def test_return_annotation_tuple_type():
    TI32 = TensorType(i32, ())

    main_op = func.FuncOp("main", ((TI32,), (TI32,)))
    with ImplicitBuilder(main_op.body) as (arg,):
        res = stablehlo.AbsOp(arg).result
        func.ReturnOp(res)

    module = ModuleOp([main_op])
    executable = JaxExecutable.compile(module)

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "Return annotation is must be jnp.ndarray or a tuple of jnp.ndarray, got tuple[int]."
        ),
    ):

        @executable  # pyright: ignore[reportArgumentType]
        def abs_wrong_tuple_type(a: jax.Array) -> tuple[int]: ...  # pyright: ignore[reportUnusedFunction]


def test_return_annotation_single():
    TI32 = TensorType(i32, ())

    main_op = func.FuncOp("main", ((TI32,), (TI32,)))
    with ImplicitBuilder(main_op.body) as (arg,):
        res = stablehlo.AbsOp(arg).result
        func.ReturnOp(res)

    module = ModuleOp([main_op])
    executable = JaxExecutable.compile(module)

    with pytest.raises(
        NotImplementedError,
        match="Return annotation is must be jnp.ndarray or a tuple of jnp.ndarray",
    ):

        @executable  # pyright: ignore[reportArgumentType]
        def abs_wrong_single_type(a: jax.Array) -> int: ...  # pyright: ignore[reportUnusedFunction]


def test_return_value_count_mismatch():
    TI32 = TensorType(i32, ())

    main_op = func.FuncOp("main", ((TI32,), (TI32, TI32)))
    with ImplicitBuilder(main_op.body) as (arg,):
        res = stablehlo.AbsOp(arg).result
        func.ReturnOp(res, res)

    module = ModuleOp([main_op])
    executable = JaxExecutable.compile(module)

    with pytest.raises(
        ValueError,
        match="Number of return values .* does not match the stub's return annotation",
    ):

        @executable
        def abs_return_count_mismatch(a: jax.Array) -> jax.Array: ...  # pyright: ignore[reportUnusedFunction]
