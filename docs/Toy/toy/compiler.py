from io import StringIO
from pathlib import Path


from xdsl.ir import MLContext
from xdsl.dialects.builtin import ModuleOp, Builtin
from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation
from xdsl.transforms.lower_riscv_func import LowerRISCVFunc

from xdsl.interpreters.riscv_emulator import run_riscv

from .frontend.ir_gen import IRGen
from .frontend.parser import Parser

from .rewrites.lower_toy import LowerToy
from .rewrites.optimise_toy import OptimiseToy
from .rewrites.lower_vector import LowerVector
from .rewrites.optimise_vector import OptimiseVector
from .rewrites.setup_riscv_pass import SetupRiscvPass
from .rewrites.lower_llvm import LowerLLVM


from .emulator.toy_accelerator_instructions import ToyAccelerator

from .dialects import toy, vector
from xdsl.dialects import riscv, riscv_func, cf, scf, printf


def context() -> MLContext:
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(toy.Toy)
    ctx.register_dialect(vector.Vector)
    ctx.register_dialect(riscv.RISCV)
    ctx.register_dialect(riscv_func.RISCV_Func)
    ctx.register_dialect(cf.Cf)
    ctx.register_dialect(scf.Scf)
    ctx.register_dialect(printf.Printf)
    return ctx


def parse_toy(program: str, ctx: MLContext | None = None) -> ModuleOp:
    mlir_gen = IRGen()
    module_ast = Parser(Path("in_memory"), program).parseModule()
    module_op = mlir_gen.ir_gen_module(module_ast)
    return module_op


def compile(program: str) -> str:
    ctx = context()

    op = parse_toy(program)

    OptimiseToy().apply(ctx, op)
    LowerToy().apply(ctx, op)
    OptimiseVector().apply(ctx, op)
    LowerVector().apply(ctx, op)
    SetupRiscvPass().apply(ctx, op)
    LowerLLVM().apply(ctx, op)
    LowerRISCVFunc(insert_exit_syscall=True).apply(ctx, op)
    RISCVRegisterAllocation().apply(ctx, op)

    io = StringIO()
    riscv.print_assembly(op, io)

    return io.getvalue()


def emulate_riscv(program: str):
    run_riscv(program, extensions=[ToyAccelerator], unlimited_regs=True, verbosity=0)
