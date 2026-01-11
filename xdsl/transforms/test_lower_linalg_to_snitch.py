from dataclasses import dataclass

from xdsl.backend.riscv.lowering import (
    convert_arith_to_riscv,
    convert_arith_to_riscv_snitch,
    convert_func_to_riscv_func,
    convert_memref_to_riscv,
    convert_riscv_scf_to_riscv_cf,
    convert_scf_to_riscv_scf,
    convert_snitch_stream_to_snitch,
)
from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.transforms import (
    canonicalize,
    convert_linalg_to_memref_stream,
    convert_memref_stream_to_loops,
    convert_memref_stream_to_snitch_stream,
    convert_riscv_scf_for_to_frep,
    lower_affine,
    lower_snitch,
    memref_stream_fold_fill,
    memref_stream_generalize_fill,
    memref_stream_infer_fill,
    memref_stream_interleave,
    memref_stream_legalize,
    memref_stream_tile_outer_loops,
    memref_stream_unnest_out_parameters,
    memref_streamify,
    reconcile_unrealized_casts,
    riscv_allocate_registers,
    riscv_lower_parallel_mov,
    riscv_scf_loop_range_folding,
    scf_for_loop_flatten,
    snitch_allocate_registers,
)

OPTIMISE_MEMREF_STREAM_PASSES: tuple[ModulePass, ...] = (
    canonicalize.CanonicalizePass(),
    memref_stream_infer_fill.MemRefStreamInferFillPass(),
    memref_stream_unnest_out_parameters.MemRefStreamUnnestOutParametersPass(),
    memref_stream_fold_fill.MemRefStreamFoldFillPass(),
    memref_stream_generalize_fill.MemRefStreamGeneralizeFillPass(),
    memref_stream_interleave.MemRefStreamInterleavePass(),
    memref_stream_tile_outer_loops.MemRefStreamTileOuterLoopsPass(target_rank=4),
    memref_streamify.MemRefStreamifyPass(),
    convert_memref_stream_to_loops.ConvertMemRefStreamToLoopsPass(),
    canonicalize.CanonicalizePass(),
    scf_for_loop_flatten.ScfForLoopFlattenPass(),
)

LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES: tuple[ModulePass, ...] = (
    canonicalize.CanonicalizePass(),
    convert_memref_to_riscv.ConvertMemRefToRiscvPass(),
    lower_affine.LowerAffinePass(),
    convert_scf_to_riscv_scf.ConvertScfToRiscvPass(),
    convert_arith_to_riscv_snitch.ConvertArithToRiscvSnitchPass(),
    convert_arith_to_riscv.ConvertArithToRiscvPass(),
    convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass(),
    convert_memref_stream_to_snitch_stream.ConvertMemRefStreamToSnitchStreamPass(),
    reconcile_unrealized_casts.ReconcileUnrealizedCastsPass(),
)

LOWER_SNITCH_STREAM_TO_ASM_PASSES: tuple[ModulePass, ...] = (
    canonicalize.CanonicalizePass(),
    convert_riscv_scf_for_to_frep.ConvertRiscvScfForToFrepPass(),
    snitch_allocate_registers.SnitchAllocateRegistersPass(),
    convert_snitch_stream_to_snitch.ConvertSnitchStreamToSnitch(),
    lower_snitch.LowerSnitchPass(),
    canonicalize.CanonicalizePass(),
    riscv_scf_loop_range_folding.RiscvScfLoopRangeFoldingPass(),
    canonicalize.CanonicalizePass(),
    riscv_allocate_registers.RISCVAllocateRegistersPass(add_regalloc_stats=True),
    canonicalize.CanonicalizePass(),
    convert_riscv_scf_to_riscv_cf.ConvertRiscvScfToRiscvCfPass(),
    canonicalize.CanonicalizePass(),
)

TEST_LOWER_LINALG_TO_SNITCH_PASSES: tuple[ModulePass, ...] = (
    canonicalize.CanonicalizePass(),
    convert_linalg_to_memref_stream.ConvertLinalgToMemRefStreamPass(),
    memref_stream_legalize.MemRefStreamLegalizePass(),
    *OPTIMISE_MEMREF_STREAM_PASSES,
    *LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES,
    *LOWER_SNITCH_STREAM_TO_ASM_PASSES,
    riscv_lower_parallel_mov.RISCVLowerParallelMovPass(),
    canonicalize.CanonicalizePass(),
)


@dataclass(frozen=True)
class TestLowerLinalgToSnitchPass(ModulePass):
    """
    A compiler pass used for testing lowering microkernels from linalg generic to snitch
    assembly.
    """

    name = "test-lower-linalg-to-snitch"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for p in TEST_LOWER_LINALG_TO_SNITCH_PASSES:
            p.apply(ctx, op)
