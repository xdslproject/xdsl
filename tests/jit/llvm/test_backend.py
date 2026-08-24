import ctypes

import pytest

pytest.importorskip("llvmlite.binding")

from xdsl.context import Context
from xdsl.dialects import func, llvm
from xdsl.dialects.builtin import ModuleOp
from xdsl.jit.function import RawJITFunc
from xdsl.jit.llvm.backend import LLVMJITBackend, LLVMRawJITFunc
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


def test_backend_registers_llvm_ctypes():
    # the builtin ctypes are already covered by the jit test
    c_type_context = LLVMJITBackend().c_type_context
    assert c_type_context.to_ctype(llvm.LLVMPointerType()) is ctypes.c_void_p


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
