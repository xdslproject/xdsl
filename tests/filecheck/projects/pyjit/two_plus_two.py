# RUN: python %s | filecheck %s

from collections.abc import Callable
from ctypes import CFUNCTYPE
from dataclasses import dataclass
from typing import Any, Generic, ParamSpec

import llvmlite
import llvmlite.binding
import llvmlite.ir as llvm_ir
from typing_extensions import TypeVar

from xdsl.backend.llvm.convert import convert_module
from xdsl.dialects import arith, builtin, func, llvm
from xdsl.dialects.builtin import ModuleOp
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.jit.llvm.convert_ctypes import CTypeContext, register_builtin_ctypes
from xdsl.traits import SymbolTable
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


def mcjit_compile(
    xdsl_module: ModuleOp,
    symbol: str,
    ctype_ctx: CTypeContext,
) -> McJitKeepalive[..., Any]:
    llvm_module: llvm_ir.Module = convert_module(xdsl_module)
    llvm_ir_text = str(llvm_module)
    llvmlite.binding.initialize_native_target()  # pyright: ignore
    llvmlite.binding.initialize_native_asmprinter()  # pyright: ignore

    target = llvmlite.binding.Target.from_default_triple()  # pyright: ignore
    target_machine = target.create_target_machine()  # pyright: ignore
    backing_mod = llvmlite.binding.parse_assembly(llvm_ir_text)  # pyright: ignore
    engine = llvmlite.binding.create_mcjit_compiler(backing_mod, target_machine)  # pyright: ignore
    engine.finalize_object()  # pyright: ignore
    engine.run_static_constructors()  # pyright: ignore

    func_op = SymbolTable.lookup_symbol(xdsl_module, symbol)
    assert isinstance(func_op, llvm.FuncOp)
    ret_ctype = ctype_ctx.to_ctype(func_op.function_type.output)
    arg_ctypes = [ctype_ctx.to_ctype(t) for t in func_op.function_type.inputs]
    fn_type = CFUNCTYPE(ret_ctype, *arg_ctypes)

    func_ptr = engine.get_function_address(symbol)  # pyright: ignore
    fn = fn_type(func_ptr)  # pyright: ignore

    return McJitKeepalive(
        target=target,  # pyright: ignore
        target_machine=target_machine,  # pyright: ignore
        backing_mod=backing_mod,  # pyright: ignore
        engine=engine,  # pyright: ignore
        func=fn,
    )


# JIT


# TODO: support extending the JIT with more functionality
class JITContext:
    pyast_ctx: PyASTContext
    ctype_ctx: CTypeContext

    def __init__(self):
        ctx = PyASTContext(post_transforms=[FrontendDesymrefyPass(), convert_to_llvm])
        ctx.register_type(float, builtin.f64)
        ctx.register_function(float.__add__, arith.AddfOp)
        ctx.register_dialect(arith.Arith)
        ctx.register_dialect(llvm.LLVM)
        ctx.register_dialect(builtin.Builtin)
        ctx.register_dialect(func.Func)
        self.pyast_ctx = ctx

        self.ctype_ctx = CTypeContext()
        register_builtin_ctypes(self.ctype_ctx)

    def jit(self, func: Callable[P, R]) -> McJitKeepalive[P, R]:
        parsed_program = self.pyast_ctx.parse_program(func)
        return mcjit_compile(parsed_program.module, parsed_program.name, self.ctype_ctx)


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
