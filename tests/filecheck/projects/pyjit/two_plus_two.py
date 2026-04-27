# RUN: python %s | filecheck %s

from collections.abc import Callable
from ctypes import CFUNCTYPE, c_double
from dataclasses import dataclass
from typing import Generic, ParamSpec

import llvmlite
import llvmlite.binding
import llvmlite.ir as llvm_ir
from typing_extensions import TypeVar

from xdsl.backend.llvm.convert import convert_module
from xdsl.dialects import arith, builtin, func, llvm
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.transforms.desymref import FrontendDesymrefyPass
from xdsl.transforms.mlir_opt import MLIROptPass

# Lowering

# TODO: add passes to xDSL
convert_to_llvm = MLIROptPass(
    arguments=("--convert-arith-to-llvm", "--convert-func-to-llvm"),
    generic=True,
)

# Executable

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(slots=True)
class McJitKeepalive(Generic[P, R]):
    """Holds LLVM MCJIT-owned objects so jitted code is not unmapped by GC."""

    target: object
    target_machine: object
    backing_mod: object
    engine: object
    func: Callable[P, R]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.func(*args, **kwargs)


# TODO: support automatic conversion of types
def mcjit_f64_f64_f64_binary(
    llvm_module: llvm_ir.Module, symbol: str
) -> McJitKeepalive[[float, float], float]:
    llvm_ir_text = str(llvm_module)
    llvmlite.binding.initialize_native_target()  # pyright: ignore
    llvmlite.binding.initialize_native_asmprinter()  # pyright: ignore

    target = llvmlite.binding.Target.from_default_triple()  # pyright: ignore
    target_machine = target.create_target_machine()  # pyright: ignore
    backing_mod = llvmlite.binding.parse_assembly(llvm_ir_text)  # pyright: ignore
    engine = llvmlite.binding.create_mcjit_compiler(backing_mod, target_machine)  # pyright: ignore
    engine.finalize_object()  # pyright: ignore
    engine.run_static_constructors()  # pyright: ignore

    func_ptr = engine.get_function_address(symbol)  # pyright: ignore
    fn_type = CFUNCTYPE(c_double, c_double, c_double)
    fn = fn_type(func_ptr)  # pyright: ignore

    keepalive = McJitKeepalive(
        target=target,  # pyright: ignore
        target_machine=target_machine,  # pyright: ignore
        backing_mod=backing_mod,  # pyright: ignore
        engine=engine,  # pyright: ignore
        func=fn,
    )

    return keepalive


# JIT


# TODO: support extending the JIT with more functionality
class JITContext:
    pyast_ctx: PyASTContext

    def __init__(self):
        ctx = PyASTContext(post_transforms=[FrontendDesymrefyPass(), convert_to_llvm])
        ctx.register_type(float, builtin.f64)
        ctx.register_function(float.__add__, arith.AddfOp)
        ctx.register_dialect(arith.Arith)
        ctx.register_dialect(llvm.LLVM)
        ctx.register_dialect(builtin.Builtin)
        ctx.register_dialect(func.Func)
        self.pyast_ctx = ctx

    def jit(
        self, func: Callable[[float, float], float]
    ) -> McJitKeepalive[[float, float], float]:
        parsed_program = self.pyast_ctx.parse_program(func)
        module = convert_module(parsed_program.module)
        return mcjit_f64_f64_f64_binary(module, parsed_program.name)


# Test

ctx = JITContext()


@ctx.jit
def plus(a: float, b: float) -> float:
    return a + b


# CHECK: plus(2.0, 2.0) = 4.0
# CHECK: plus(3.0, 4.0) = 7.0
print(f"{plus(2.0, 2.0) = }")
print(f"{plus(3.0, 4.0) = }")

# CHECK: plus.func(2.0, 2.0) = 4.0
# CHECK: plus.func(3.0, 4.0) = 7.0
print(f"{plus.func(2.0, 2.0) = }")
print(f"{plus.func(3.0, 4.0) = }")
