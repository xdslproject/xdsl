import pytest

try:
    from jax import make_jaxpr  # pyright: ignore[reportUnknownVariableType]
    from jax import numpy as jnp
except ImportError as exc:
    print(exc)
    pytest.skip("jax is an optional dependency", allow_module_level=True)

from xdsl.frontend.jaxpr import IRGen

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
