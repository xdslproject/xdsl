# RUN: python %s | filecheck %s

import jax
import jax.numpy as jnp
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
            "stablehlo-aggressive-simplification,"
            "stablehlo-legalize-to-linalg"
            "))"
        )

        pm.run(module.operation)

        return str(module)


key = jax.random.key(42)


def matadd(A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray):
    return A + B


# print(get_linalg_str(jax.jit(matadd, donate_argnames="C", keep_unused=True), (jax.random.uniform(key, [8, 16], dtype=np.float64), jax.random.uniform(key, [8, 16], dtype=np.float64), jax.random.uniform(key, [8, 16], dtype=np.float64))))

changed = """
#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8x16xf64>, %arg1: tensor<8x16xf64>, %arg2: tensor<8x16xf64> {tf.aliasing_output = 0 : i32}) -> tensor<8x16xf64> {
    %0 = tensor.empty() : tensor<8x16xf64>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x16xf64>, tensor<8x16xf64>) outs(%0 : tensor<8x16xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %2 = arith.addf %in, %in_0 : f64
      linalg.yield %2 : f64
    } -> tensor<8x16xf64>
    return %1 : tensor<8x16xf64>
  }
}
"""

print(changed)
