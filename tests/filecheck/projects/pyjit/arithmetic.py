# RUN: python %s | filecheck %s

"""
JIT compilation of Python functions via xDSL IR.

Frontend builds mid-level IR. Backend lowers to its dialect and binds ctypes
from IR types. Context only parses, delegates, and wraps Python values.

Library authors configure a :class:`JITContext` with a frontend
(:class:`~xdsl.frontend.pyast.context.PyASTContext`), ctypes bridges, and a
:class:`JITBackend`. End users apply :meth:`JITContext.jit` without knowing
about compilers::

    @ctx.jit(Callable[[float, float], float])
    def plus(a: float, b: float) -> float:
        return a + b

Pipeline:

1. Parse the Python function to a ``ModuleOp`` (PyAST), applying frontend
   post-transforms.
2. Hand the module to the backend, which lowers to its dialect and produces a
   :class:`RawJITFunc`.
3. Wrap it as a :class:`WrappedJITFunc` that marshals Python values through ctypes.

The ``Callable[...]`` argument to :meth:`JITContext.jit` is the ABI signature used
for marshalling. It is passed explicitly so annotations need not be evaluated.
"""

from collections.abc import Callable
from dataclasses import dataclass

import llvmlite
import llvmlite.binding
import llvmlite.ir as llvm_ir

from xdsl.backend.llvm.convert import convert_module
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, llvm
from xdsl.jit.c_type_context import register_builtin_ctypes
from xdsl.jit.context import JITBackend, JITContext, register_builtin_type_maps
from xdsl.jit.function import CFunc, CFuncType, RawJITFunc
from xdsl.jit.llvm.c_type_context import register_llvm_ctypes, to_c_func_type
from xdsl.passes import ModulePass, PassPipeline
from xdsl.traits import SymbolTable
from xdsl.transforms.mlir_opt import MLIROptPass

# --- LLVM / llvmlite backend (xdsl.jit.llvm) ---


@dataclass(slots=True, init=False)
class LLVMRawJITFunc(RawJITFunc):
    """
    :class:`RawJITFunc` that retains LLVM MCJIT runtime objects.

    The engine and related objects must remain referenced for as long as
    ``c_func`` may be called.
    """

    target: object
    target_machine: object
    backing_mod: object
    engine: object

    def __init__(
        self,
        c_func_type: CFuncType,
        c_func: CFunc,
        target: object,
        target_machine: object,
        backing_mod: object,
        engine: object,
    ):
        super(LLVMRawJITFunc, self).__init__(c_func_type, c_func)
        self.target = target
        self.target_machine = target_machine
        self.backing_mod = backing_mod
        self.engine = engine


def llvm_jit(
    llvm_module: llvm_ir.Module, symbol: str, c_func_type: CFuncType
) -> LLVMRawJITFunc:
    """Compile ``llvm_module`` with MCJIT and bind ``symbol`` to ``c_func_type``."""
    llvm_ir_text = str(llvm_module)
    llvmlite.binding.initialize_native_target()
    llvmlite.binding.initialize_native_asmprinter()

    target = llvmlite.binding.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvmlite.binding.parse_assembly(llvm_ir_text)
    engine = llvmlite.binding.create_mcjit_compiler(backing_mod, target_machine)
    engine.finalize_object()
    engine.run_static_constructors()

    func_ptr = engine.get_function_address(symbol)
    c_types_fn = c_func_type(func_ptr)

    keepalive = LLVMRawJITFunc(
        c_func_type,
        c_types_fn,
        target=target,
        target_machine=target_machine,
        backing_mod=backing_mod,
        engine=engine,
    )

    return keepalive


class LLVMJITBackend(JITBackend):
    """
    :class:`JITBackend` using xDSL’s LLVM converter and llvmlite MCJIT.

    Runs :attr:`lowering`, requires ``symbol`` to name an ``llvm.FuncOp``, then
    converts the module and JITs it.
    """

    lowering: tuple[ModulePass, ...]
    """Pass pipeline applied before resolving ``symbol``."""

    def __init__(
        self,
        lowering: tuple[ModulePass, ...] = (
            MLIROptPass(
                arguments=("--convert-arith-to-llvm", "--convert-func-to-llvm"),
                generic=True,
            ),
        ),
    ):
        """Construct the backend with the given ``lowering`` pipeline."""
        super().__init__()
        register_builtin_ctypes(self.c_type_context)
        register_llvm_ctypes(self.c_type_context)
        self.lowering = lowering

    def jit(
        self,
        mlir_module: builtin.ModuleOp,
        symbol: str,
        ir_context: Context,
    ) -> RawJITFunc:
        """Lower ``mlir_module``, bind ``symbol``, and return an :class:`LLVMRawJITFunc`."""
        ir_context.load_dialect(llvm.LLVM)
        PassPipeline(self.lowering).apply(ir_context, mlir_module)
        func_op = SymbolTable.lookup_symbol(mlir_module, symbol)
        assert isinstance(func_op, llvm.FuncOp)
        c_func_type = to_c_func_type(self.c_type_context, func_op.function_type)
        llvm_module = convert_module(mlir_module, fallback_target_triple=None)
        return llvm_jit(llvm_module, symbol, c_func_type)


# --- Example: library-author configuration ---

ctx = JITContext(LLVMJITBackend())

ctx.pyast_ctx.register_function(float.__add__, arith.AddfOp)
ctx.pyast_ctx.register_dialect(arith.Arith)
ctx.pyast_ctx.register_dialect(builtin.Builtin)
ctx.pyast_ctx.register_dialect(func.Func)

register_builtin_type_maps(ctx)

# --- Example: end-user code ---


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
