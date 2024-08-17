import jax
import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import func, stablehlo
from xdsl.dialects.builtin import ModuleOp, TensorType, i32

pytest.importorskip("jax")

from xdsl.backend.jax_jit import jax_jit  # noqa: E402


def test_abs():
    TI32 = TensorType(i32, ())

    main_op = func.FuncOp("main", ((TI32,), (TI32,)))
    with ImplicitBuilder(main_op.body) as (arg,):
        res = stablehlo.AbsOp(arg).result
        func.Return(res)

    module = ModuleOp([main_op])

    lazyjit = jax_jit(module)

    @lazyjit
    def abs_tuple(a: jax.Array) -> tuple[jax.Array]: ...

    assert abs_tuple(jax.numpy.array(-2, dtype=jax.numpy.int32))[0] == jax.numpy.array(
        2, dtype=jax.numpy.int32
    )
    assert abs_tuple(jax.numpy.array(0, dtype=jax.numpy.int32))[0] == jax.numpy.array(
        0, dtype=jax.numpy.int32
    )
    assert abs_tuple(jax.numpy.array(2, dtype=jax.numpy.int32))[0] == jax.numpy.array(
        2, dtype=jax.numpy.int32
    )

    @lazyjit
    def abs_one(a: jax.Array) -> jax.Array: ...

    assert abs_one(jax.numpy.array(-2, dtype=jax.numpy.int32)) == jax.numpy.array(
        2, dtype=jax.numpy.int32
    )
    assert abs_one(jax.numpy.array(0, dtype=jax.numpy.int32)) == jax.numpy.array(
        0, dtype=jax.numpy.int32
    )
    assert abs_one(jax.numpy.array(2, dtype=jax.numpy.int32)) == jax.numpy.array(
        2, dtype=jax.numpy.int32
    )


def test_no_main():
    module = ModuleOp([])

    with pytest.raises(ValueError, match="No `main` function in module"):
        jax_jit(module)


def test_fail():
    TI32 = TensorType(i32, ())

    with pytest.raises(ValueError, match="No `main` function in module"):
        main_op = func.FuncOp("not_main", ((TI32,), (TI32,)))
        with ImplicitBuilder(main_op.body) as (arg,):
            res = stablehlo.AbsOp(arg).result
            func.Return(res)

        module = ModuleOp([main_op])

        @jax_jit(module)
        def abs(a: jax.Array) -> tuple[jax.Array]: ...
