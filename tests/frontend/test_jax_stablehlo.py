import pytest

try:
    import jax
    import jax.numpy as jnp
    from jax._src.interpreters import mlir
    from jaxlib.mlir.dialects import (
        stablehlo,  # pyright: ignore[reportMissingTypeStubs]
    )
    from jaxlib.mlir.ir import Context, Module
    from jaxlib.mlir.passmanager import PassManager
except ImportError as exc:
    print(exc)
    pytest.skip("jax is an optional dependency", allow_module_level=True)


def get_linalg_str(func_jit, args):
    lowered = func_jit.lower(*args)
    module = lowered.compiler_ir(dialect="stablehlo")
    module_str = str(module)

    with Context() as ctx:
        ctx.append_dialect_registry(mlir.upstream_dialects)
        stablehlo.register_dialect(ctx)
        stablehlo.register_stablehlo_passes()

        module = Module.parse(module_str)

        pm = PassManager.parse(
            "builtin.module(func.func("
            "shape-legalize-to-stablehlo,"
            "stablehlo-aggressive-folder,"
            "stablehlo-aggressive-simplification,"
            "stablehlo-legalize-to-linalg"
            "))"
        )

        pm.run(module.operation)

        return str(module)


def test_matmul():
    def matmul(A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray):
        return A @ B

    key = jax.random.key(42)

    compiled_string = get_linalg_str(
        jax.jit(matmul, donate_argnames="C", keep_unused=True),
        (
            jax.random.uniform(key, [8, 8]),
            jax.random.uniform(key, [8, 8]),
            jax.random.uniform(key, [8, 8]),
        ),
    )
    print(compiled_string)
    expected_string = """module @jit_matmul attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32> {tf.aliasing_output = 0 : i32}) -> (tensor<8x8xf32> {jax.result_info = ""}) {
    %0 = tensor.empty() : tensor<8x8xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%1 : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %2 : tensor<8x8xf32>
  }
}
"""

    assert compiled_string == expected_string
