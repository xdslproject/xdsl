# RUN: python %s | filecheck %s

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.interpreters import mlir
from jaxlib.mlir.dialects import stablehlo
from jaxlib.mlir.ir import Context, Module
from jaxlib.mlir.passmanager import PassManager

jax.config.update("jax_enable_x64", True)


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
            "stablehlo-aggressive-simplification"
            # "stablehlo-legalize-to-linalg"
            "))"
        )

        pm.run(module.operation)

        return str(module)


key = jax.random.key(42)


def matadd(A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray):
    return A + B


# breaks because of memref<f64>
def dot(x: jnp.ndarray, y: jnp.ndarray):
    return jnp.dot(x, y)


def matmul(A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray):
    return A @ B


def relu(A: jnp.ndarray, B: jnp.ndarray):
    return jnp.maximum(A, 0)


# breaks because of memref<f64>
def fill(val: np.float64, A: jnp.ndarray):
    return jnp.full(A.shape, val)


def conv(A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray):
    return lax.conv_general_dilated(A, B, (1, 1), "VALID")


# print(get_linalg_str(jax.jit(matadd, donate_argnames="C", keep_unused=True), (jax.random.uniform(key, [8, 16], dtype=np.float64), jax.random.uniform(key, [8, 16], dtype=np.float64), jax.random.uniform(key, [8, 16], dtype=np.float64))))

# print(get_linalg_str(jax.jit(dot), (jax.random.uniform(key, [100], dtype=np.float64), jax.random.uniform(key, [100], dtype=np.float64))))

# print(get_linalg_str(jax.jit(matmul, donate_argnames="C", keep_unused=True), (jax.random.uniform(key, [8, 8], dtype=np.float64), jax.random.uniform(key, [8, 8], dtype=np.float64), jax.random.uniform(key, [8, 8], dtype=np.float64))))

# print(get_linalg_str(jax.jit(relu, donate_argnames="B", keep_unused=True), (jax.random.uniform(key, [16, 16], dtype=np.float64), jax.random.uniform(key, [16, 16], dtype=np.float64))))

# print(get_linalg_str(jax.jit(fill, donate_argnames="A", keep_unused=True), (150., jax.random.uniform(key, [16, 16], dtype=np.float64))))

print(
    get_linalg_str(
        jax.jit(conv, donate_argnames="C", keep_unused=True),
        (
            jax.random.uniform(key, [1, 1, 10, 10], dtype=np.float64),
            jax.random.uniform(key, [1, 1, 3, 3], dtype=np.float64),
            jax.random.uniform(key, [1, 1, 8, 8], dtype=np.float64),
        ),
    )
)

changed = """
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6, d0, d2 + d3, d4 + d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d0, d3, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6, d1, d2, d4)>
module attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x10x10xf64>, %arg1: tensor<1x1x3x3xf64>, %arg2: tensor<1x1x8x8xf64> {tf.aliasing_output = 0 : i32}) -> tensor<1x1x8x8xf64> {
    %0 = tensor.empty() : tensor<1x1x3x3xf64>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x1x3x3xf64>) outs(%0 : tensor<1x1x3x3xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<1x1x3x3xf64>
    %2 = tensor.empty() : tensor<1x1x8x8xf64>
    %cst = arith.constant 0.000000e+00 : f64
    %3 = linalg.fill ins(%cst : f64) outs(%2 : tensor<1x1x8x8xf64>) -> tensor<1x1x8x8xf64>
    %4 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "reduction", "parallel"]} ins(%arg0, %1 : tensor<1x1x10x10xf64>, tensor<1x1x3x3xf64>) outs(%3 : tensor<1x1x8x8xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %5 = arith.mulf %in, %in_0 : f64
      %6 = arith.addf %out, %5 : f64
      linalg.yield %6 : f64
    } -> tensor<1x1x8x8xf64>
    %collapsed = tensor.collapse_shape %4 [[0, 1, 2, 3]] : tensor<1x1x8x8xf64> into tensor<64xf64>
    %cast = tensor.cast %collapsed : tensor<64xf64> to tensor<64xf64>
    %expanded = tensor.expand_shape %cast [[0, 1, 2, 3]] output_shape [1, 1, 8, 8] : tensor<64xf64> into tensor<1x1x8x8xf64>
    return %expanded : tensor<1x1x8x8xf64>
  }
}
"""

# print(changed)
