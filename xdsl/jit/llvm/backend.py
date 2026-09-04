from dataclasses import dataclass
from typing import cast

import llvmlite.binding
import llvmlite.ir as llvm_ir
from cffi import FFI
from llvmlite.binding.executionengine import ExecutionEngine
from llvmlite.binding.module import ModuleRef
from llvmlite.binding.targets import Target, TargetMachine

from xdsl.backend.llvm.convert import convert_module
from xdsl.context import Context
from xdsl.dialects import builtin, llvm
from xdsl.jit.c_type_context import CFuncSignature, register_builtin_types
from xdsl.jit.context import JITBackend
from xdsl.jit.function import CFunc, RawJITFunc
from xdsl.jit.llvm.c_type_context import register_llvm_types, to_c_func_type
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


_FFI = FFI()


def _create_target_machine(*, opt_level: int) -> tuple[Target, TargetMachine]:
    llvmlite.binding.initialize_native_target()
    llvmlite.binding.initialize_native_asmprinter()

    target = llvmlite.binding.Target.from_triple(llvmlite.binding.get_process_triple())
    target_machine = target.create_target_machine(
        cpu=llvmlite.binding.get_host_cpu_name(),
        features=llvmlite.binding.get_host_cpu_features().flatten(),
        opt=opt_level,
        jit=True,
    )
    return target, target_machine


def _compile_module(
    llvm_module: llvm_ir.Module,
    symbol: str,
    c_func_type: CFuncSignature,
    *,
    target: Target,
    target_machine: TargetMachine,
    opt_level: int,
) -> LLVMRawJITFunc:
    backing_mod = llvmlite.binding.parse_assembly(str(llvm_module))
    if backing_mod.triple not in (
        "",
        "unknown-unknown-unknown",
        target.triple,
        target_machine.triple,
    ):
        raise JITException(
            f"Cannot JIT module for target {backing_mod.triple} with native "
            f"target {target_machine.triple}"
        )
    backing_mod.triple = target_machine.triple
    backing_mod.data_layout = str(target_machine.target_data)

    options = llvmlite.binding.PipelineTuningOptions(speed_level=opt_level)
    options.slp_vectorization = True
    with (
        options,
        llvmlite.binding.create_pass_builder(target_machine, options) as pass_builder,
        pass_builder.getModulePassManager() as module_pass_manager,
    ):
        module_pass_manager.run(backing_mod, pass_builder)

    engine = llvmlite.binding.create_mcjit_compiler(backing_mod, target_machine)
    engine.finalize_object()
    engine.run_static_constructors()

    func_ptr = engine.get_function_address(symbol)
    if not func_ptr:
        # MCJIT reports an unresolved symbol as a null address rather than raising
        raise JITException(f"No address for symbol after compilation: {symbol}")
    params = ", ".join(c_func_type.inputs) or "void"
    signature = f"{c_func_type.output}(*)({params})"
    cffi_func = cast(CFunc, _FFI.cast(signature, func_ptr))

    return LLVMRawJITFunc(
        c_func_type,
        cffi_func,
        target=target,
        target_machine=target_machine,
        backing_mod=backing_mod,
        engine=engine,
    )


class LLVMJITBackend(JITBackend):
    """
    :class:`JITBackend` using xDSL's LLVM converter and llvmlite MCJIT.

    Runs :attr:`lowering`, requires ``symbol`` to name an ``llvm.FuncOp``, then
    converts the module and exposes the entry point through CFFI ABI mode.
    """

    lowering: tuple[ModulePass, ...]
    """Pass pipeline applied before resolving ``symbol``."""

    opt_level: int
    """LLVM optimization level, from 0 to 3, applied to codegen and the IR pipeline."""

    def __init__(
        self,
        lowering: tuple[ModulePass, ...] = (
            MLIROptPass(
                arguments=("--convert-arith-to-llvm", "--convert-func-to-llvm"),
                generic=True,
            ),
        ),
        *,
        opt_level: int = 2,
    ):
        """Construct the backend with the given ``lowering`` and ``opt_level``."""
        if opt_level not in range(4):
            raise ValueError(f"opt_level must be between 0 and 3, got {opt_level}")
        super().__init__()
        register_builtin_types(self.c_type_context)
        register_llvm_types(self.c_type_context)
        self.lowering = lowering
        self.opt_level = opt_level

    def jit(
        self,
        mlir_module: builtin.ModuleOp,
        symbol: str,
        ir_context: Context,
    ) -> LLVMRawJITFunc:
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
        target, target_machine = _create_target_machine(opt_level=self.opt_level)
        llvm_module = convert_module(
            mlir_module,
            fallback_target_triple=target_machine.triple,
            data_layout=str(target_machine.target_data),
        )
        return _compile_module(
            llvm_module,
            symbol,
            c_func_type,
            target=target,
            target_machine=target_machine,
            opt_level=self.opt_level,
        )
