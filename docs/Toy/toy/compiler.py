from io import StringIO
from pathlib import Path

from xdsl.backend.riscv.lowering.convert_arith_to_riscv import ConvertArithToRiscvPass
from xdsl.backend.riscv.lowering.convert_func_to_riscv_func import (
    ConvertFuncToRiscvFuncPass,
)
from xdsl.backend.riscv.lowering.convert_memref_to_riscv import ConvertMemRefToRiscvPass
from xdsl.backend.riscv.lowering.convert_print_format_to_riscv_debug import (
    ConvertPrintFormatToRiscvDebugPass,
)
from xdsl.backend.riscv.lowering.convert_riscv_scf_to_riscv_cf import (
    ConvertRiscvScfToRiscvCfPass,
)
from xdsl.backend.riscv.lowering.convert_scf_to_riscv_scf import ConvertScfToRiscvPass
from xdsl.context import Context
from xdsl.dialects import (
    affine,
    arith,
    func,
    memref,
    printf,
    riscv,
    riscv_func,
    scf,
)
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.dead_code_elimination import DeadCodeElimination
from xdsl.transforms.lower_affine import LowerAffinePass
from xdsl.transforms.lower_riscv_func import LowerRISCVFunc
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl.transforms.riscv_allocate_registers import RISCVAllocateRegistersPass
from xdsl.transforms.riscv_scf_loop_range_folding import RiscvScfLoopRangeFoldingPass
from xdsl.transforms.shape_inference import ShapeInferencePass

from .dialects import toy
from .frontend.ir_gen import IRGen
from .frontend.parser import ToyParser
from .rewrites.inline_toy import InlineToyPass
from .rewrites.lower_toy import LowerToyPass


def context() -> Context:
    ctx = Context()
    ctx.load_dialect(affine.Affine)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(printf.Printf)
    ctx.load_dialect(riscv_func.RISCV_Func)
    ctx.load_dialect(riscv.RISCV)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(toy.Toy)
    return ctx


def parse_toy(program: str, ctx: Context | None = None) -> ModuleOp:
    mlir_gen = IRGen()
    module_ast = ToyParser(Path("in_memory"), program).parse_module()
    module_op = mlir_gen.ir_gen_module(module_ast)
    return module_op


def transform(
    ctx: Context,
    module_op: ModuleOp,
    *,
    target: str = "riscv-assembly",
):
    if target == "toy":
        return

    CanonicalizePass().apply(ctx, module_op)

    if target == "toy-opt":
        return

    InlineToyPass().apply(ctx, module_op)

    if target == "toy-inline":
        return

    ShapeInferencePass().apply(ctx, module_op)

    if target == "shape-inference":
        return

    LowerToyPass().apply(ctx, module_op)
    module_op.verify()

    if target == "affine":
        return

    LowerAffinePass().apply(ctx, module_op)

    if target == "scf":
        return

    ConvertFuncToRiscvFuncPass().apply(ctx, module_op)
    ConvertMemRefToRiscvPass().apply(ctx, module_op)
    ConvertPrintFormatToRiscvDebugPass().apply(ctx, module_op)
    ConvertArithToRiscvPass().apply(ctx, module_op)
    ConvertScfToRiscvPass().apply(ctx, module_op)
    DeadCodeElimination().apply(ctx, module_op)
    ReconcileUnrealizedCastsPass().apply(ctx, module_op)

    module_op.verify()

    if target == "riscv":
        return

    # Perform optimizations that don't depend on register allocation
    # e.g. constant folding
    CanonicalizePass().apply(ctx, module_op)
    RiscvScfLoopRangeFoldingPass().apply(ctx, module_op)
    CanonicalizePass().apply(ctx, module_op)

    module_op.verify()

    if target == "riscv-opt":
        return

    RISCVAllocateRegistersPass(allow_infinite=True).apply(ctx, module_op)

    module_op.verify()

    if target == "riscv-regalloc":
        return

    # Perform optimizations that depend on register allocation
    # e.g. redundant moves
    CanonicalizePass().apply(ctx, module_op)

    module_op.verify()

    if target == "riscv-regalloc-opt":
        return

    LowerRISCVFunc(insert_exit_syscall=True).apply(ctx, module_op)
    ConvertRiscvScfToRiscvCfPass().apply(ctx, module_op)

    if target == "riscv-lowered":
        return

    raise ValueError(f"Unknown target option {target}")


def compile(program: str) -> str:
    ctx = context()

    op = parse_toy(program)
    transform(ctx, op, target="riscv-lowered")

    io = StringIO()
    riscv.print_assembly(op, io)

    return io.getvalue()


def emulate_riscv(program: str):
    from xdsl.interpreters.riscv_emulator import run_riscv

    run_riscv(program, unlimited_regs=True, verbosity=0)
