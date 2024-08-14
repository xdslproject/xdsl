import jax
import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import func, stablehlo
from xdsl.dialects.builtin import ModuleOp, TensorType, i32

pytest.importorskip("jax")

from xdsl.backend.jax import jax_jit  # noqa: E402


def test_no_main():
    module = ModuleOp([])

    with pytest.raises(ValueError, match="No `main` function in module."):
        jax_jit(module)


def test_abs():
    TI32 = TensorType(i32, ())

    main_op = func.FuncOp("main", ((TI32,), (TI32,)))
    with ImplicitBuilder(main_op.body) as (arg,):
        res = stablehlo.AbsOp(arg).result
        func.Return(res)

    module = ModuleOp([main_op])

    @jax_jit(module)
    def abs(a: jax.Array) -> jax.Array: ...

    assert abs(jax.numpy.array(-2, dtype=jax.numpy.int32)) == jax.numpy.array(
        2, dtype=jax.numpy.int32
    )
    assert abs(jax.numpy.array(0, dtype=jax.numpy.int32)) == jax.numpy.array(
        0, dtype=jax.numpy.int32
    )
    assert abs(jax.numpy.array(2, dtype=jax.numpy.int32)) == jax.numpy.array(
        2, dtype=jax.numpy.int32
    )
