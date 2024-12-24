# RUN: python %s | mlir-opt --split-input-file  --allow-unregistered-dialect --linalg-generalize-named-ops | xdsl-opt --split-input-file  -p jax-use-donated-arguments | \
# RUN: mlir-opt --split-input-file  --allow-unregistered-dialect --eliminate-empty-tensors --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --canonicalize --mlir-print-op-generic | \
# RUN: xdsl-opt --split-input-file  -p test-lower-linalg-to-snitch -t riscv-asm | filecheck %s

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


# CHECK:       .text
# CHECK-NEXT:  .globl main
# CHECK-NEXT:  .p2align 2
# CHECK-NEXT:      # Regalloc stats: {"preallocated_float": ["ft0", "ft1", "ft2"], "preallocated_int": ["a0", "a1", "a2", "zero"], "allocated_float": ["ft0", "ft1", "ft2"], "allocated_int": ["a0", "a1", "a2", "t0", "t1", "t2", "t3", "zero"]}
# CHECK-NEXT:  main:
# CHECK-NEXT:      mv t2, a0
# CHECK-NEXT:      mv t1, a1
# CHECK-NEXT:      mv t0, a2
# CHECK-NEXT:      li t3, 127
# CHECK-NEXT:      scfgwi t3, 95                                # dm 31 dim 0 bound
# CHECK-NEXT:      li t3, 8
# CHECK-NEXT:      scfgwi t3, 223                               # dm 31 dim 0 stride
# CHECK-NEXT:      scfgwi zero, 63                              # dm 31 repeat
# CHECK-NEXT:      scfgwi t2, 768                               # dm 0 dim 0 source
# CHECK-NEXT:      scfgwi t1, 769                               # dm 1 dim 0 source
# CHECK-NEXT:      scfgwi t0, 898                               # dm 2 dim 0 destination
# CHECK-NEXT:      csrrsi zero, 1984, 1                         # SSR enable
# CHECK-NEXT:      li t1, 127
# CHECK-NEXT:      frep.o t1, 1, 0, 0
# CHECK-NEXT:      fadd.d ft2, ft0, ft1
# CHECK-NEXT:      csrrci zero, 1984, 1                         # SSR disable
# CHECK-NEXT:      mv a0, t0
# CHECK-NEXT:      ret
def matadd(A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray):
    return A + B


print(
    get_linalg_str(
        jax.jit(matadd, donate_argnames="C", keep_unused=True),
        (
            jax.random.uniform(key, [8, 16], dtype=np.float64),
            jax.random.uniform(key, [8, 16], dtype=np.float64),
            jax.random.uniform(key, [8, 16], dtype=np.float64),
        ),
    )
)
print("// -----")


# CHECK:       .text
# CHECK-NEXT:  .globl main
# CHECK-NEXT:  .p2align 2
# CHECK-NEXT:      # Regalloc stats: {{.*}}
# CHECK-NEXT:  main:
# CHECK-NEXT:      mv t2, a0
# CHECK-NEXT:      mv t3, a1
# CHECK-NEXT:      mv t0, a2
# CHECK-NEXT:      fcvt.d.w ft3, zero
# CHECK-NEXT:      li t1, 7
# CHECK-NEXT:      scfgwi t1, 64                                # dm 0 dim 0 bound
# CHECK-NEXT:      li t1, 1
# CHECK-NEXT:      scfgwi t1, 96                                # dm 0 dim 1 bound
# CHECK-NEXT:      li t1, 7
# CHECK-NEXT:      scfgwi t1, 128                               # dm 0 dim 2 bound
# CHECK-NEXT:      li t1, 8
# CHECK-NEXT:      scfgwi t1, 192                               # dm 0 dim 0 stride
# CHECK-NEXT:      li t1, -56
# CHECK-NEXT:      scfgwi t1, 224                               # dm 0 dim 1 stride
# CHECK-NEXT:      li t1, 8
# CHECK-NEXT:      scfgwi t1, 256                               # dm 0 dim 2 stride
# CHECK-NEXT:      li t1, 3
# CHECK-NEXT:      scfgwi t1, 32                                # dm 0 repeat
# CHECK-NEXT:      li t1, 3
# CHECK-NEXT:      scfgwi t1, 65                                # dm 1 dim 0 bound
# CHECK-NEXT:      li t1, 7
# CHECK-NEXT:      scfgwi t1, 97                                # dm 1 dim 1 bound
# CHECK-NEXT:      li t1, 1
# CHECK-NEXT:      scfgwi t1, 129                               # dm 1 dim 2 bound
# CHECK-NEXT:      li t1, 7
# CHECK-NEXT:      scfgwi t1, 161                               # dm 1 dim 3 bound
# CHECK-NEXT:      li t1, 8
# CHECK-NEXT:      scfgwi t1, 193                               # dm 1 dim 0 stride
# CHECK-NEXT:      li t1, 40
# CHECK-NEXT:      scfgwi t1, 225                               # dm 1 dim 1 stride
# CHECK-NEXT:      li t1, -440
# CHECK-NEXT:      scfgwi t1, 257                               # dm 1 dim 2 stride
# CHECK-NEXT:      li t1, -504
# CHECK-NEXT:      scfgwi t1, 289                               # dm 1 dim 3 stride
# CHECK-NEXT:      scfgwi zero, 33                              # dm 1 repeat
# CHECK-NEXT:      li t1, 63
# CHECK-NEXT:      scfgwi t1, 66                                # dm 2 dim 0 bound
# CHECK-NEXT:      li t1, 8
# CHECK-NEXT:      scfgwi t1, 194                               # dm 2 dim 0 stride
# CHECK-NEXT:      scfgwi zero, 34                              # dm 2 repeat
# CHECK-NEXT:      scfgwi t2, 832                               # dm 0 dim 2 source
# CHECK-NEXT:      scfgwi t3, 865                               # dm 1 dim 3 source
# CHECK-NEXT:      scfgwi t0, 898                               # dm 2 dim 0 destination
# CHECK-NEXT:      csrrsi zero, 1984, 1                         # SSR enable
# CHECK-NEXT:      li t2, 16
# CHECK-NEXT:      mv t1, zero
# CHECK-NEXT:      # Constant folded riscv_cf.bge
# CHECK-NEXT:  scf_body_0_for:
# CHECK-NEXT:      fmv.d ft7, ft3
# CHECK-NEXT:      fmv.d ft6, ft3
# CHECK-NEXT:      fmv.d ft5, ft3
# CHECK-NEXT:      fmv.d ft4, ft3
# CHECK-NEXT:      li t4, 7
# CHECK-NEXT:      frep.o t4, 8, 0, 0
# CHECK-NEXT:      fmul.d ft11, ft0, ft1
# CHECK-NEXT:      fmul.d ft10, ft0, ft1
# CHECK-NEXT:      fmul.d ft9, ft0, ft1
# CHECK-NEXT:      fmul.d ft8, ft0, ft1
# CHECK-NEXT:      fadd.d ft7, ft7, ft11
# CHECK-NEXT:      fadd.d ft6, ft6, ft10
# CHECK-NEXT:      fadd.d ft5, ft5, ft9
# CHECK-NEXT:      fadd.d ft4, ft4, ft8
# CHECK-NEXT:      fmv.d ft2, ft7
# CHECK-NEXT:      fmv.d ft2, ft6
# CHECK-NEXT:      fmv.d ft2, ft5
# CHECK-NEXT:      fmv.d ft2, ft4
# CHECK-NEXT:      addi t1, t1, 1
# CHECK-NEXT:      blt t1, t2, scf_body_0_for
# CHECK-NEXT:  scf_body_end_0_for:
# CHECK-NEXT:      csrrci zero, 1984, 1                         # SSR disable
# CHECK-NEXT:      mv a0, t0
# CHECK-NEXT:      ret
def matmul(A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray):
    return A @ B


print(
    get_linalg_str(
        jax.jit(matmul, donate_argnames="C", keep_unused=True),
        (
            jax.random.uniform(key, [8, 8], dtype=np.float64),
            jax.random.uniform(key, [8, 8], dtype=np.float64),
            jax.random.uniform(key, [8, 8], dtype=np.float64),
        ),
    )
)
print("// -----")


# CHECK:        .text
# CHECK-NEXT:   .globl main
# CHECK-NEXT:   .p2align 2
# CHECK-NEXT:       # Regalloc stats: {"preallocated_float": ["ft0", "ft1", "ft2"], "preallocated_int": ["a0", "a1", "zero"], "allocated_float": ["ft0", "ft1", "ft3"], "allocated_int": ["a0", "a1", "t0", "t1", "t2", "zero"]}
# CHECK-NEXT:   main:
# CHECK-NEXT:       mv t1, a0
# CHECK-NEXT:       mv t0, a1
# CHECK-NEXT:       fcvt.d.w ft3, zero
# CHECK-NEXT:       li t2, 255
# CHECK-NEXT:       scfgwi t2, 95                                # dm 31 dim 0 bound
# CHECK-NEXT:       li t2, 8
# CHECK-NEXT:       scfgwi t2, 223                               # dm 31 dim 0 stride
# CHECK-NEXT:       scfgwi zero, 63                              # dm 31 repeat
# CHECK-NEXT:       scfgwi t1, 768                               # dm 0 dim 0 source
# CHECK-NEXT:       scfgwi t0, 897                               # dm 1 dim 0 destination
# CHECK-NEXT:       csrrsi zero, 1984, 1                         # SSR enable
# CHECK-NEXT:       li t1, 255
# CHECK-NEXT:       frep.o t1, 1, 0, 0
# CHECK-NEXT:       fmax.d ft1, ft0, ft3
# CHECK-NEXT:       csrrci zero, 1984, 1                         # SSR disable
# CHECK-NEXT:       mv a0, t0
# CHECK-NEXT:       ret
def relu(A: jnp.ndarray, B: jnp.ndarray):
    return jnp.maximum(A, 0)


print(
    get_linalg_str(
        jax.jit(relu, donate_argnames="B", keep_unused=True),
        (
            jax.random.uniform(key, [16, 16], dtype=np.float64),
            jax.random.uniform(key, [16, 16], dtype=np.float64),
        ),
    )
)


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
