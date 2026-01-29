from xdsl.context import Context
from xdsl.interpreter import Interpreter
from xdsl.interpreters import (
    affine,
    arith,
    builtin,
    cf,
    func,
    linalg,
    memref,
    memref_stream,
    ml_program,
    pdl,
    printf,
    riscv,
    riscv_cf,
    riscv_debug,
    riscv_func,
    riscv_libc,
    riscv_scf,
    riscv_snitch,
    rv32,
    rv64,
    scf,
    snitch_stream,
    tensor,
)


def register_implementations(interpreter: Interpreter, ctx: Context):
    interpreter.register_implementations(affine.AffineFunctions())
    interpreter.register_implementations(arith.ArithFunctions())
    interpreter.register_implementations(builtin.BuiltinFunctions())
    interpreter.register_implementations(cf.CfFunctions())
    interpreter.register_implementations(func.FuncFunctions())
    interpreter.register_implementations(linalg.LinalgFunctions())
    interpreter.register_implementations(memref_stream.MemRefStreamFunctions())
    interpreter.register_implementations(memref.MemRefFunctions())
    interpreter.register_implementations(ml_program.MLProgramFunctions())
    interpreter.register_implementations(pdl.PDLRewriteFunctions(ctx))
    interpreter.register_implementations(printf.PrintfFunctions())
    interpreter.register_implementations(riscv_cf.RiscvCfFunctions())
    interpreter.register_implementations(riscv_debug.RiscvDebugFunctions())
    interpreter.register_implementations(riscv_func.RiscvFuncFunctions())
    interpreter.register_implementations(riscv_libc.RiscvLibcFunctions())
    interpreter.register_implementations(riscv_scf.RiscvScfFunctions())
    interpreter.register_implementations(riscv_snitch.RiscvSnitchFunctions())
    interpreter.register_implementations(riscv.RiscvFunctions())
    interpreter.register_implementations(rv32.Rv32())
    interpreter.register_implementations(rv64.Rv64())
    interpreter.register_implementations(scf.ScfFunctions())
    interpreter.register_implementations(snitch_stream.SnitchStreamFunctions())
    interpreter.register_implementations(tensor.TensorFunctions())
