from typing import Literal

import pytest
from cffi import FFI

pytest.importorskip("llvmlite.binding")

import llvmlite.binding as llvm_binding
import llvmlite.ir as llvm_ir

from xdsl.context import Context
from xdsl.dialects import func, llvm
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.jit.c_type_context import CFuncSignature
from xdsl.jit.function import RawJITFunc
from xdsl.jit.llvm.backend import (
    LLVMJITBackend,
    LLVMRawJITFunc,
    _compile_module,  # pyright: ignore[reportPrivateUsage]
    _create_target_machine,  # pyright: ignore[reportPrivateUsage]
)
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import JITException


def parse(module: str) -> ModuleOp:
    ctx = Context()
    ctx.load_dialect(llvm.LLVM)
    ctx.load_dialect(func.Func)
    return Parser(ctx, module).parse_module()


def jit(
    module: str, symbol: str = "plus", lowering: tuple[ModulePass, ...] = ()
) -> RawJITFunc:
    # the test modules are already lowered, so an empty pipeline keeps mlir-opt out
    return LLVMJITBackend(lowering=lowering).jit(parse(module), symbol, Context())


PLUS = """
llvm.func @plus(%a: f64, %b: f64) -> f64 {
  %0 = llvm.fadd %a, %b : f64
  llvm.return %0 : f64
}
"""


def test_jit_compiles_and_binds_symbol():
    raw_func = jit(PLUS)
    # the subclass keeps the MCJIT engine alive while `c_func` is callable
    assert isinstance(raw_func, LLVMRawJITFunc)
    assert raw_func.c_func(2.0, 2.0) == 4.0
    assert raw_func.c_func(3.0, 4.0) == 7.0


POINTER_IDENTITY = """
llvm.func @identity(%p: !llvm.ptr) -> !llvm.ptr {
  llvm.return %p : !llvm.ptr
}
"""


def test_jit_pointer_call():
    raw_func = jit(POINTER_IDENTITY, "identity")
    pointer = FFI().cast("void *", 1)
    assert raw_func.c_func(pointer) == pointer


def test_jit_is_reusable_across_calls():
    # a JITContext reuses one ir_context for every function it decorates
    backend = LLVMJITBackend(lowering=())
    ir_context = Context()
    for _ in range(2):
        raw_func = backend.jit(parse(PLUS), "plus", ir_context)
        assert raw_func.c_func(2.0, 2.0) == 4.0


def test_lowering_is_applied():
    lowered: list[ModuleOp] = []

    class RecordingPass(ModulePass):
        name = "recording"

        def apply(self, ctx: Context, op: ModuleOp) -> None:
            lowered.append(op)

    jit(PLUS, lowering=(RecordingPass(),))
    assert len(lowered) == 1


def test_backend_registers_llvm_types():
    c_type_context = LLVMJITBackend().c_type_context
    assert c_type_context.to_c_type(llvm.LLVMPointerType()) == "void *"


def test_missing_symbol_raises():
    with pytest.raises(JITException, match="No symbol to JIT compile: absent"):
        jit(PLUS, "absent")


def test_unlowered_symbol_raises():
    with pytest.raises(JITException, match="must leave it in the LLVM dialect"):
        jit("func.func private @plus(f64, f64) -> f64")


def test_declaration_without_body_raises():
    # a declaration resolves and converts, but MCJIT has no code to bind
    with pytest.raises(JITException, match="No address for symbol"):
        jit("llvm.func @plus(f64, f64) -> f64")


IDENTITY = """
llvm.func @identity(%value: i64) -> i64 {
  llvm.return %value : i64
}
"""


ADD_ZERO = """
llvm.func @add_zero(%value: i64) -> i64 {
  %zero = llvm.mlir.constant(0 : i64) : i64
  %result = llvm.add %value, %zero : i64
  llvm.return %result : i64
}
"""


@pytest.mark.parametrize("opt_level", [1, 2, 3])
def test_pipeline_optimizes_module(opt_level: Literal[1, 2, 3]):
    optimized = LLVMJITBackend(lowering=(), opt_level=opt_level).jit(
        parse(ADD_ZERO), "add_zero", Context()
    )

    assert " add i64 " not in str(optimized.backing_mod)
    assert optimized.c_func(41) == 41


def test_default_opt_level():
    assert LLVMJITBackend(lowering=()).opt_level == 2


def test_pipeline_does_not_optimize_module():
    unoptimized = LLVMJITBackend(lowering=(), opt_level=0).jit(
        parse(ADD_ZERO), "add_zero", Context()
    )

    assert " add i64 " in str(unoptimized.backing_mod)


def test_jit_rejects_non_native_target_triple():
    incompatible_arch = (
        "aarch64"
        if llvm_binding.get_process_triple().startswith("x86_64")
        else "x86_64"
    )
    module = parse(IDENTITY)
    module.attributes["llvm.target_triple"] = StringAttr(
        f"{incompatible_arch}-unknown-unknown"
    )

    with pytest.raises(JITException, match="Cannot JIT module for target"):
        LLVMJITBackend(lowering=()).jit(module, "identity", Context())


def identity_module(*, linkage: str = "") -> llvm_ir.Module:
    module = llvm_ir.Module()
    func_type = llvm_ir.FunctionType(llvm_ir.IntType(64), (llvm_ir.IntType(64),))
    function = llvm_ir.Function(module, func_type, "identity")
    function.linkage = linkage
    builder = llvm_ir.IRBuilder(function.append_basic_block())
    builder.ret(function.args[0])
    return module


def compile_module(
    module: llvm_ir.Module, symbol: str, c_func_type: CFuncSignature
) -> LLVMRawJITFunc:
    # the converter only emits native modules, so these guards need llvmlite IR
    target, target_machine = _create_target_machine(opt_level=2)
    return _compile_module(
        module,
        symbol,
        c_func_type,
        target=target,
        target_machine=target_machine,
        opt_level=2,
    )


def test_jit_rejects_non_native_data_layout():
    module = identity_module()
    module.data_layout = "e-p:32:32"

    with pytest.raises(JITException, match="non-native data layout"):
        compile_module(module, "identity", CFuncSignature(("int64_t",), "int64_t"))


def test_jit_rejects_non_function_symbol():
    module = llvm_ir.Module()
    llvm_ir.GlobalVariable(module, llvm_ir.IntType(64), "value")

    with pytest.raises(JITException, match="No function to JIT compile: value"):
        compile_module(module, "value", CFuncSignature((), "int64_t"))


def test_optimization_preserves_requested_symbol():
    # global DCE drops an internal entry point unless the backend exports it
    raw_func = compile_module(
        identity_module(linkage="internal"),
        "identity",
        CFuncSignature(("int64_t",), "int64_t"),
    )

    assert raw_func.c_func(42) == 42
