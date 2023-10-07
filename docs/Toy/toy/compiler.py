from collections.abc import Iterator
from io import StringIO
from pathlib import Path

from xdsl.backend.riscv.lowering.convert_arith_to_riscv import ConvertArithToRiscvPass
from xdsl.backend.riscv.lowering.convert_func_to_riscv_func import (
    ConvertFuncToRiscvFuncPass,
)
from xdsl.backend.riscv.lowering.convert_memref_to_riscv import ConvertMemrefToRiscvPass
from xdsl.backend.riscv.lowering.convert_scf_to_riscv_scf import ConvertScfToRiscvPass
from xdsl.backend.riscv.lowering.reduce_register_pressure import (
    RiscvReduceRegisterPressurePass,
)
from xdsl.backend.riscv.riscv_scf_to_asm import (
    LowerScfForToLabels,
)
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
from xdsl.interpreters.riscv_emulator import run_riscv
from xdsl.ir import MLContext
from xdsl.passes import ModulePass, PassPipelinePass
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.dead_code_elimination import DeadCodeElimination
from xdsl.transforms.lower_affine import LowerAffinePass
from xdsl.transforms.lower_riscv_func import LowerRISCVFunc
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation
from xdsl.transforms.riscv_scf_loop_range_folding import RiscvScfLoopRangeFoldingPass

from .dialects import toy
from .emulator.toy_accelerator_instructions import ToyAccelerator
from .frontend.ir_gen import IRGen
from .frontend.parser import Parser
from .rewrites.inline_toy import InlineToyPass
from .rewrites.lower_linalg_stream import LinalgToStreamPass
from .rewrites.lower_memref_riscv import LowerMemrefToRiscv
from .rewrites.lower_printf_riscv import LowerPrintfRiscvPass
from .rewrites.lower_stream_affine import StreamToAffinePass
from .rewrites.lower_toy_linalg import LowerToLinalgPass
from .rewrites.setup_riscv_pass import SetupRiscvPass
from .rewrites.shape_inference import ShapeInferencePass


def context() -> MLContext:
    ctx = MLContext()
    ctx.register_dialect(affine.Affine)
    ctx.register_dialect(arith.Arith)
    ctx.register_dialect(Builtin)
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


def _toy_passes() -> Iterator[ModulePass]:
    return iter(())


def _toy_opt_passes() -> Iterator[ModulePass]:
    yield from _toy_passes()
    yield CanonicalizePass()


def _toy_inline_passes() -> Iterator[ModulePass]:
    yield from _toy_opt_passes()
    yield InlineToyPass()


def _toy_infer_shapes_passes() -> Iterator[ModulePass]:
    yield from _toy_inline_passes()
    yield ShapeInferencePass()


def _linalg_passes() -> Iterator[ModulePass]:
    yield from _toy_infer_shapes_passes()
    yield LowerToLinalgPass()


def _stream_passes() -> Iterator[ModulePass]:
    yield from _linalg_passes()
    yield LinalgToStreamPass()


def _affine_passes() -> Iterator[ModulePass]:
    yield from _stream_passes()
    yield StreamToAffinePass()


def _scf_passes() -> Iterator[ModulePass]:
    yield from _affine_passes()
    yield LowerAffinePass()


def _riscv_passes() -> Iterator[ModulePass]:
    yield from _scf_passes()
    yield SetupRiscvPass()
    yield ConvertFuncToRiscvFuncPass()
    yield LowerMemrefToRiscv()
    yield ConvertMemrefToRiscvPass()
    yield LowerPrintfRiscvPass()
    yield ConvertArithToRiscvPass()
    yield ConvertScfToRiscvPass()
    yield DeadCodeElimination()
    yield ReconcileUnrealizedCastsPass()


def _riscv_opt_passes() -> Iterator[ModulePass]:
    yield from _riscv_passes()

    # Perform optimizations that don't depend on register allocation
    # e.g. constant folding
    yield CanonicalizePass()
    yield RiscvScfLoopRangeFoldingPass()
    yield CanonicalizePass()
    yield RiscvReduceRegisterPressurePass()


def _riscv_regalloc_passes() -> Iterator[ModulePass]:
    yield from _riscv_opt_passes()

    yield RISCVRegisterAllocation()


def _riscv_regalloc_opt_passes() -> Iterator[ModulePass]:
    yield from _riscv_regalloc_passes()

    # Perform optimizations that depend on register allocation
    # e.g. redundant moves
    yield CanonicalizePass()


def _riscv_lowered_passes() -> Iterator[ModulePass]:
    yield from _riscv_regalloc_opt_passes()

    yield LowerRISCVFunc(insert_exit_syscall=True)
    yield LowerScfForToLabels()


def pass_pipeline(target: str) -> PassPipelinePass:
    generators = {
        "toy": _toy_passes,
        "toy-opt": _toy_opt_passes,
        "toy-inline": _toy_inline_passes,
        "toy-infer-shapes": _toy_infer_shapes_passes,
        "linalg": _linalg_passes,
        "stream": _stream_passes,
        "affine": _affine_passes,
        "scf": _scf_passes,
        "riscv": _riscv_passes,
        "riscv-opt": _riscv_opt_passes,
        "riscv-regalloc": _riscv_regalloc_passes,
        "riscv-regalloc-opt": _riscv_regalloc_opt_passes,
        "riscv-lowered": _riscv_lowered_passes,
    }

    if target not in generators:
        raise ValueError(f"Unknown target option {target}")
    return PassPipelinePass(list(generators[target]()))


def transform(
    ctx: MLContext,
    module_op: ModuleOp,
    *,
    target: str = "riscv-assembly",
):
    pass_pipeline(target).apply(ctx, module_op)


def compile(program: str) -> str:
    ctx = context()

    op = parse_toy(program)
    transform(ctx, op, target="riscv-lowered")

    io = StringIO()
    riscv.print_assembly(op, io)

    return io.getvalue()


def emulate_riscv(program: str):
    run_riscv(program, extensions=[ToyAccelerator], unlimited_regs=True, verbosity=0)
