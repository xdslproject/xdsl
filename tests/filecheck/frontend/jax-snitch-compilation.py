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


def matmul(A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray):
    return A @ B


# print(get_linalg_str(jax.jit(matmul, donate_argnames="C", keep_unused=True), (jax.random.uniform(key, [8, 8], dtype=np.float64), jax.random.uniform(key, [8, 8], dtype=np.float64), jax.random.uniform(key, [8, 8], dtype=np.float64))))


def relu(A: jnp.ndarray, B: jnp.ndarray):
    return jnp.maximum(A, 0)


# print(get_linalg_str(jax.jit(relu, donate_argnames="B", keep_unused=True), (jax.random.uniform(key, [16, 16], dtype=np.float64), jax.random.uniform(key, [16, 16], dtype=np.float64))))


# breaks because of memref<f64>
def dot(x: jnp.ndarray, y: jnp.ndarray):
    return jnp.dot(x, y)


# print(get_linalg_str(jax.jit(dot), (jax.random.uniform(key, [100], dtype=np.float64), jax.random.uniform(key, [100], dtype=np.float64))))


# breaks because of memref<f64>
def fill(val: np.float64, A: jnp.ndarray):
    return jnp.full(A.shape, val)


# print(get_linalg_str(jax.jit(fill, donate_argnames="A", keep_unused=True), (150., jax.random.uniform(key, [16, 16], dtype=np.float64))))


# a weird copy is inserted
def conv(A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray):
    return lax.conv_general_dilated(A, B, (1, 1), "VALID")


# print(get_linalg_str(jax.jit(conv, donate_argnames="C", keep_unused=True), (jax.random.uniform(key, [1, 1, 10, 10], dtype=np.float64),jax.random.uniform(key, [1, 1, 3, 3], dtype=np.float64),jax.random.uniform(key, [1, 1, 8, 8], dtype=np.float64),),))


# one of the reduction dimensions is f32 => it can't be streamed and it breaks
def max_pool(A: jnp.ndarray, B: jnp.ndarray):
    return lax.reduce_window(A, -10000.0, lax.max, [1, 1, 3, 3], [1, 1, 2, 2], "VALID")


# print(get_linalg_str(jax.jit(max_pool, donate_argnames="B", keep_unused=True), (jax.random.uniform(key, [1, 1, 18, 18], dtype=np.float64), jax.random.uniform(key, [1, 1, 8, 8], dtype=np.float64))))
