from xdsl.interpreter import Interpreter
from xdsl.interpreters import (
    affine,
    arith,
    builtin,
    cf,
    func,
    linalg,
    memref,
    printf,
    riscv,
    riscv_debug,
    riscv_func,
    riscv_libc,
    riscv_snitch,
    scf,
    snitch_stream,
)
from xdsl.interpreters.experimental import pdl
from xdsl.ir import MLContext


def register_implementations(
    interpreter: Interpreter,
    ctx: MLContext,
    include_wgpu: bool = True,
):
    interpreter.register_implementations(func.FuncFunctions())
    interpreter.register_implementations(cf.CfFunctions())
    interpreter.register_implementations(riscv.RiscvFunctions())
    interpreter.register_implementations(riscv_debug.RiscvDebugFunctions())
    interpreter.register_implementations(riscv_func.RiscvFuncFunctions())
    interpreter.register_implementations(riscv_libc.RiscvLibcFunctions())
    interpreter.register_implementations(riscv_snitch.RiscvSnitchFunctions())
    interpreter.register_implementations(pdl.PDLRewriteFunctions(ctx))
    interpreter.register_implementations(affine.AffineFunctions())
    interpreter.register_implementations(linalg.LinalgFunctions())
    interpreter.register_implementations(memref.MemrefFunctions())
    if include_wgpu:
        from xdsl.interpreters.experimental import wgpu

        interpreter.register_implementations(wgpu.WGPUFunctions())
    interpreter.register_implementations(builtin.BuiltinFunctions())
    interpreter.register_implementations(arith.ArithFunctions())
    interpreter.register_implementations(printf.PrintfFunctions())
    interpreter.register_implementations(scf.ScfFunctions())
    interpreter.register_implementations(snitch_stream.SnitchStreamFunctions())
