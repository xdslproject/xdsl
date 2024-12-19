# RUN: python %s | filecheck %s

import jax
import jax.numpy as jnp
from jax._src.interpreters import mlir
from jaxlib.mlir.dialects import stablehlo
from jaxlib.mlir.ir import Context, Module
from jaxlib.mlir.passmanager import PassManager


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


key = jax.random.key(42)


def scale(x: jnp.ndarray, alpha: float):
    return x * alpha


# print(get_linalg_str(jax.jit(scale), (jax.random.uniform(key, [10]), 0.1)))

original = """
#map = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
module @jit_scale attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<10xf32>, %arg1: tensor<f32>) -> (tensor<10xf32> {jax.result_info = ""}) {
    %0 = tensor.empty() : tensor<10xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%arg1 : tensor<f32>) outs(%0 : tensor<10xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<10xf32>
    %2 = tensor.empty() : tensor<10xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %1 : tensor<10xf32>, tensor<10xf32>) outs(%2 : tensor<10xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.mulf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<10xf32>
    return %3 : tensor<10xf32>
  }
}
"""

changed = """
#map = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
module attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<10xf32>, %arg1: tensor<f32>) -> tensor<10xf32> {
    %0 = tensor.empty() : tensor<10xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%arg1 : tensor<f32>) outs(%0 : tensor<10xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<10xf32>
    %2 = tensor.empty() : tensor<10xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %1 : tensor<10xf32>, tensor<10xf32>) outs(%2 : tensor<10xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.mulf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<10xf32>
    return %3 : tensor<10xf32>
  }
}
"""

print(changed)
