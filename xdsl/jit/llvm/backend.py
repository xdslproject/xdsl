from dataclasses import dataclass

import llvmlite
import llvmlite.binding
import llvmlite.ir as llvm_ir
from llvmlite.binding.executionengine import ExecutionEngine
from llvmlite.binding.module import ModuleRef
from llvmlite.binding.targets import Target, TargetMachine

from xdsl.backend.llvm.convert import convert_module
from xdsl.context import Context
from xdsl.dialects import builtin, llvm
from xdsl.jit.c_type_context import register_builtin_ctypes
from xdsl.jit.context import JITBackend
from xdsl.jit.function import CFuncType, RawJITFunc
from xdsl.jit.llvm.c_type_context import register_llvm_ctypes, to_c_func_type
from xdsl.passes import ModulePass, PassPipeline
from xdsl.traits import SymbolTable
from xdsl.transforms.mlir_opt import MLIROptPass
from xdsl.utils.exceptions import JITException


@dataclass(slots=True)
class LLVMRawJITFunc(RawJITFunc):
    """
    :class:`RawJITFunc` that retains LLVM MCJIT runtime objects.

    The engine and related objects must remain referenced for as long as
    ``c_func`` may be called.
    """

    target: Target
    target_machine: TargetMachine
    backing_mod: ModuleRef
    engine: ExecutionEngine


def llvm_jit(
    llvm_module: llvm_ir.Module, symbol: str, c_func_type: CFuncType
) -> LLVMRawJITFunc:
    """Compile ``llvm_module`` with MCJIT and bind ``symbol`` to ``c_func_type``."""
    llvmlite.binding.initialize_native_target()
    llvmlite.binding.initialize_native_asmprinter()

    target = llvmlite.binding.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvmlite.binding.parse_assembly(str(llvm_module))
    engine = llvmlite.binding.create_mcjit_compiler(backing_mod, target_machine)
    engine.finalize_object()
    engine.run_static_constructors()

    func_ptr = engine.get_function_address(symbol)
    if not func_ptr:
        # MCJIT reports an unresolved symbol as a null address rather than raising
        raise JITException(f"No address for symbol after compilation: {symbol}")
    c_types_fn = c_func_type(func_ptr)

    return LLVMRawJITFunc(
        c_func_type,
        c_types_fn,
        target=target,
        target_machine=target_machine,
        backing_mod=backing_mod,
        engine=engine,
    )


class LLVMJITBackend(JITBackend):
    """
    :class:`JITBackend` using xDSL's LLVM converter and llvmlite MCJIT.

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
        # `jit` may be called more than once against the same context
        if llvm.LLVM.name not in ir_context.registered_dialect_names:
            ir_context.load_dialect(llvm.LLVM)
        PassPipeline(self.lowering).apply(ir_context, mlir_module)
        func_op = SymbolTable.lookup_symbol(mlir_module, symbol)
        if func_op is None:
            raise JITException(f"No symbol to JIT compile: {symbol}")
        if not isinstance(func_op, llvm.FuncOp):
            raise JITException(
                f"Symbol {symbol} is a {func_op.name}, not an llvm.func: "
                "the lowering must leave it in the LLVM dialect"
            )
        c_func_type = to_c_func_type(self.c_type_context, func_op.function_type)
        llvm_module = convert_module(mlir_module, fallback_target_triple=None)
        return llvm_jit(llvm_module, symbol, c_func_type)
