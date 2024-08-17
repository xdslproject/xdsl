import jax
import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import func, stablehlo
from xdsl.dialects.builtin import ModuleOp, TensorType, i32

pytest.importorskip("jax")

from xdsl.backend.jax_executable import JaxExecutable, array  # noqa: E402


def test_abs():
    TI32 = TensorType(i32, ())

    main_op = func.FuncOp("main", ((TI32,), (TI32,)))
    with ImplicitBuilder(main_op.body) as (arg,):
        res = stablehlo.AbsOp(arg).result
        func.Return(res)

    module = ModuleOp([main_op])

    executable = JaxExecutable.compile(module)

    @executable
    def abs_tuple(a: jax.Array) -> tuple[jax.Array]: ...

    assert abs_tuple(array(-2, dtype=jax.numpy.int32))[0] == array(
        2, dtype=jax.numpy.int32
    )
    assert abs_tuple(array(0, dtype=jax.numpy.int32))[0] == array(
        0, dtype=jax.numpy.int32
    )
    assert abs_tuple(array(2, dtype=jax.numpy.int32))[0] == array(
        2, dtype=jax.numpy.int32
    )

    @executable
    def abs_one(a: jax.Array) -> jax.Array: ...

    assert abs_one(array(-2, dtype=jax.numpy.int32)) == array(2, dtype=jax.numpy.int32)
    assert abs_one(array(0, dtype=jax.numpy.int32)) == array(0, dtype=jax.numpy.int32)
    assert abs_one(array(2, dtype=jax.numpy.int32)) == array(2, dtype=jax.numpy.int32)


def test_no_main():
    module = ModuleOp([])

    with pytest.raises(ValueError, match="No `main` function in module"):
        JaxExecutable.compile(module)


def test_fail():
    TI32 = TensorType(i32, ())

    with pytest.raises(ValueError, match="No `main` function in module"):
        main_op = func.FuncOp("not_main", ((TI32,), (TI32,)))
        with ImplicitBuilder(main_op.body) as (arg,):
            res = stablehlo.AbsOp(arg).result
            func.Return(res)

        module = ModuleOp([main_op])

        @JaxExecutable.compile(module)
        def abs(a: jax.Array) -> tuple[jax.Array]: ...  # pyright: ignore[reportUnusedFunction]
