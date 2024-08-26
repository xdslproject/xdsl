import pytest

from xdsl.frontend.jaxpr import IRGen

try:
    from jax import make_jaxpr  # pyright: ignore[reportUnknownVariableType]
    from jax import numpy as jnp
except ImportError as exc:
    print(exc)
    pytest.skip("jax is an optional dependency", allow_module_level=True)


five_ones = jnp.ones(5, dtype=jnp.float32)  # pyright: ignore[reportUnknownMemberType]


def test_id():
    def id(a: jnp.ndarray) -> jnp.ndarray:
        return a

    id_jaxpr = make_jaxpr(id)(five_ones)

    builder = IRGen()

    module_op = builder.ir_gen_module(id_jaxpr)

    assert (
        str(module_op)
        == """\
builtin.module {
  func.func public @main(%0 : tensor<5xf32>) -> tensor<5xf32> {
    func.return %0 : tensor<5xf32>
  }
}"""
    )


def test_add():
    def add(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.add(a, b)  # pyright: ignore[reportUnknownMemberType]

    id_jaxpr = make_jaxpr(add)(five_ones, five_ones)

    builder = IRGen()

    module_op = builder.ir_gen_module(id_jaxpr)

    assert (
        str(module_op)
        == """\
builtin.module {
  func.func public @main(%0 : tensor<5xf32>, %1 : tensor<5xf32>) -> tensor<5xf32> {
    %2 = "stablehlo.add"(%0, %1) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
    func.return %2 : tensor<5xf32>
  }
}"""
    )
