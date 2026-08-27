# RUN: python %s | filecheck %s

from collections.abc import Callable

from xdsl.dialects import arith, builtin, func
from xdsl.jit.context import JITContext, register_builtin_type_maps
from xdsl.jit.llvm.backend import LLVMJITBackend

ctx = JITContext(LLVMJITBackend())

ctx.pyast_ctx.register_function(float.__add__, arith.AddfOp)
ctx.pyast_ctx.register_function(float.__mul__, arith.MulfOp)
ctx.pyast_ctx.register_dialect(arith.Arith)
ctx.pyast_ctx.register_dialect(builtin.Builtin)
ctx.pyast_ctx.register_dialect(func.Func)

register_builtin_type_maps(ctx)


@ctx.jit(Callable[[float, float], float])
def plus(a: float, b: float) -> float:
    return a + b


# CHECK: plus(2.0, 2.0) = 4.0
# CHECK: plus(3.0, 4.0) = 7.0
print(f"{plus(2.0, 2.0) = }")
print(f"{plus(3.0, 4.0) = }")

# CHECK: plus.original_func(2.0, 2.0) = 4.0
# CHECK: plus.original_func(3.0, 4.0) = 7.0
print(f"{plus.original_func(2.0, 2.0) = }")
print(f"{plus.original_func(3.0, 4.0) = }")

# CHECK: plus.raw_func.c_func(2.0, 2.0) = 4.0
# CHECK: plus.raw_func.c_func(3.0, 4.0) = 7.0
print(f"{plus.raw_func.c_func(2.0, 2.0) = }")
print(f"{plus.raw_func.c_func(3.0, 4.0) = }")


@ctx.jit(Callable[[float, float, float], float])
def scale(a: float, b: float, c: float) -> float:
    return a * b + c


# CHECK: scale(3.0, 4.0, 5.0) = 17.0
# CHECK: scale(1.5, 2.0, 0.5) = 3.5
print(f"{scale(3.0, 4.0, 5.0) = }")
print(f"{scale(1.5, 2.0, 0.5) = }")
