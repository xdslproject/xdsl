from io import StringIO
from pathlib import Path

from xdsl.backend.riscv.lowering.lower_func_riscv_func import LowerFuncToRiscvFunc
from xdsl.backend.riscv.lowering.riscv_arith_lowering import RISCVLowerArith
from xdsl.backend.riscv.lowering.scf_to_riscv_scf import ScfToRiscvPass
from xdsl.dialects import (
    affine,
    arith,
    cf,
    func,
    memref,
    printf,
    riscv,
    riscv_func,
    scf,
)
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.interpreters.riscv_emulator import run_riscv
from xdsl.ir import MLContext
from xdsl.transforms.dead_code_elimination import DeadCodeElimination
from xdsl.transforms.lower_riscv_func import LowerRISCVFunc
from xdsl.transforms.mlir_opt import MLIROptPass
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation
from xdsl.transforms.riscv_scf_to_asm import LowerScfForToLabelsPass

# from xdsl.transforms.rvscf_regalloc import register_allocate_function
from .dialects import toy
from .emulator.toy_accelerator_instructions import ToyAccelerator
from .frontend.ir_gen import IRGen
from .frontend.parser import Parser
from .rewrites.arith_float_to_int import CastArithFloatToInt
from .rewrites.inline_toy import InlineToyPass
from .rewrites.lower_memref_riscv import LowerMemrefToRiscv
from .rewrites.lower_printf_riscv import LowerPrintfRiscvPass
from .rewrites.lower_to_toy_accelerator import LowerToToyAccelerator
from .rewrites.lower_toy_accelerator_to_riscv import LowerToyAccelerator
from .rewrites.lower_toy_affine import LowerToAffinePass
from .rewrites.optimise_toy import OptimiseToy
from .rewrites.setup_riscv_pass import SetupRiscvPass
from .rewrites.shape_inference import ShapeInferencePass


def context() -> MLContext:
    ctx = MLContext()
    ctx.register_dialect(affine.Affine)
    ctx.register_dialect(arith.Arith)
    ctx.register_dialect(Builtin)
    ctx.register_dialect(cf.Cf)
    ctx.register_dialect(func.Func)
    ctx.register_dialect(memref.MemRef)
    ctx.register_dialect(printf.Printf)
    ctx.register_dialect(riscv_func.RISCV_Func)
    ctx.register_dialect(riscv.RISCV)
    ctx.register_dialect(scf.Scf)
    ctx.register_dialect(toy.Toy)
    return ctx


def parse_toy(program: str, ctx: MLContext | None = None) -> ModuleOp:
    mlir_gen = IRGen()
    module_ast = Parser(Path("in_memory"), program).parseModule()
    module_op = mlir_gen.ir_gen_module(module_ast)
    return module_op


def transform(
    ctx: MLContext,
    module_op: ModuleOp,
    *,
    target: str = "riscv-assembly",
    accelerate: bool,
    skip_mlir_opt: bool = False
):
    if target == "toy":
        return

    OptimiseToy().apply(ctx, module_op)

    if target == "toy-opt":
        return

    InlineToyPass().apply(ctx, module_op)

    if target == "toy-inline":
        return

    ShapeInferencePass().apply(ctx, module_op)

    if target == "toy-infer-shapes":
        return

    LowerToAffinePass().apply(ctx, module_op)
    module_op.verify()

    if accelerate:
        LowerToToyAccelerator().apply(ctx, module_op)
        module_op.verify()

    if target == "affine":
        return

    if not skip_mlir_opt:
        MLIROptPass(
            [
                "--allow-unregistered-dialect",
                "--canonicalize",
                "--cse",
                "--lower-affine",
                "--mlir-print-op-generic",
            ]
        ).apply(ctx, module_op)

    if target == "scf":
        return

    SetupRiscvPass().apply(ctx, module_op)
    LowerFuncToRiscvFunc().apply(ctx, module_op)
    LowerToyAccelerator().apply(ctx, module_op)
    LowerMemrefToRiscv().apply(ctx, module_op)
    LowerPrintfRiscvPass().apply(ctx, module_op)
    CastArithFloatToInt().apply(ctx, module_op)
    RISCVLowerArith().apply(ctx, module_op)
    ScfToRiscvPass().apply(ctx, module_op)
    DeadCodeElimination().apply(ctx, module_op)
    ReconcileUnrealizedCastsPass().apply(ctx, module_op)

    DeadCodeElimination().apply(ctx, module_op)

    module_op.verify()

    if target == "riscv":
        return

    # for op in module_op.walk():
    #     if isinstance(op, riscv_func.FuncOp):
    #         register_allocate_function(op)
    RISCVRegisterAllocation().apply(ctx, module_op)

    if target == "riscv-regalloc":
        return

    LowerScfForToLabelsPass().apply(ctx, module_op)
    LowerRISCVFunc(insert_exit_syscall=True).apply(ctx, module_op)

    module_op.verify()


def compile(
    program: str, *, accelerate: bool, skip_mlir_opt: bool = not MLIROptPass.can_run()
) -> str:
    ctx = context()

    op = parse_toy(program)
    transform(
        ctx,
        op,
        target="riscv-assembly",
        accelerate=accelerate,
        skip_mlir_opt=skip_mlir_opt,
    )

    io = StringIO()
    riscv.print_assembly(op, io)

    return io.getvalue()


def emulate_riscv(program: str):
    run_riscv(program, extensions=[ToyAccelerator], unlimited_regs=True, verbosity=0)
