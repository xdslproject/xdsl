from dataclasses import dataclass, field

from xdsl.backend.riscv.lowering import (
    convert_arith_to_riscv,
    convert_arith_to_riscv_snitch,
    convert_func_to_riscv_func,
    convert_memref_to_riscv,
    convert_riscv_scf_to_riscv_cf,
    convert_scf_to_riscv_scf,
    convert_snitch_stream_to_snitch,
)
from xdsl.context import MLContext
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
    riscv_register_allocation,
    riscv_scf_loop_range_folding,
    scf_for_loop_flatten,
    snitch_register_allocation,
)

OPTIMISE_MEMREF_STREAM_PASSES: tuple[ModulePass, ...] = (
    canonicalize.CanonicalizePass(),
    memref_stream_infer_fill.MemrefStreamInferFillPass(),
    memref_stream_unnest_out_parameters.MemrefStreamUnnestOutParametersPass(),
    memref_stream_fold_fill.MemrefStreamFoldFillPass(),
    memref_stream_generalize_fill.MemrefStreamGeneralizeFillPass(),
    memref_stream_interleave.MemrefStreamInterleavePass(),
    memref_stream_tile_outer_loops.MemrefStreamTileOuterLoopsPass(target_rank=4),
    memref_streamify.MemrefStreamifyPass(),
    convert_memref_stream_to_loops.ConvertMemrefStreamToLoopsPass(),
    canonicalize.CanonicalizePass(),
    scf_for_loop_flatten.ScfForLoopFlattenPass(),
)

LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES: tuple[ModulePass, ...] = (
    canonicalize.CanonicalizePass(),
    convert_memref_to_riscv.ConvertMemrefToRiscvPass(),
    lower_affine.LowerAffinePass(),
    convert_scf_to_riscv_scf.ConvertScfToRiscvPass(),
    convert_arith_to_riscv_snitch.ConvertArithToRiscvSnitchPass(),
    convert_arith_to_riscv.ConvertArithToRiscvPass(),
    convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass(),
    convert_memref_stream_to_snitch_stream.ConvertMemrefStreamToSnitchStreamPass(),
    reconcile_unrealized_casts.ReconcileUnrealizedCastsPass(),
)

LOWER_SNITCH_STREAM_TO_ASM_PASSES: tuple[ModulePass, ...] = (
    canonicalize.CanonicalizePass(),
    convert_riscv_scf_for_to_frep.ConvertRiscvScfForToFrepPass(),
    snitch_register_allocation.SnitchRegisterAllocation(),
    convert_snitch_stream_to_snitch.ConvertSnitchStreamToSnitch(),
    lower_snitch.LowerSnitchPass(),
    canonicalize.CanonicalizePass(),
    riscv_scf_loop_range_folding.RiscvScfLoopRangeFoldingPass(),
    canonicalize.CanonicalizePass(),
    riscv_register_allocation.RISCVRegisterAllocation(add_regalloc_stats=True),
    canonicalize.CanonicalizePass(),
    convert_riscv_scf_to_riscv_cf.ConvertRiscvScfToRiscvCfPass(),
    canonicalize.CanonicalizePass(),
)

TEST_LOWER_LINALG_TO_SNITCH_PASSES: tuple[ModulePass, ...] = (
    canonicalize.CanonicalizePass(),
    convert_linalg_to_memref_stream.ConvertLinalgToMemrefStreamPass(),
    memref_stream_legalize.MemrefStreamLegalizePass(),
    *OPTIMISE_MEMREF_STREAM_PASSES,
    *LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES,
    *LOWER_SNITCH_STREAM_TO_ASM_PASSES,
)

LINALG_SNITCH_OPTIMIZATION_PASSES: tuple[ModulePass, ...] = (
    # + Unroll and Jam (O4)
    memref_stream_interleave.MemrefStreamInterleavePass(),
    # + FRep (O3)
    convert_riscv_scf_for_to_frep.ConvertRiscvScfForToFrepPass(),
    # + Scalar Replacement (O2)
    memref_stream_unnest_out_parameters.MemrefStreamUnnestOutParametersPass(),
    # + Streams (O1)
    memref_streamify.MemrefStreamifyPass(),
)

MAX_OPT_LEVEL = len(LINALG_SNITCH_OPTIMIZATION_PASSES)


def get_excluded_passes(
    optimization_level: int = MAX_OPT_LEVEL,
) -> tuple[ModulePass, ...]:
    """
    This function determines which optimization passes should be excluded from the
    lowering pipeline based on the specified optimization level. A higher optimization
    level includes more passes.

    Args:
        optimization_level (int): The desired optimization level, ranging from 0 to
            4 (inclusive). Defaults to 4.

    Returns:
        tuple[ModulePass, ...]: A tuple containing the ModulePass objects to be excluded
        from the lowering pipeline.
    """

    if optimization_level == MAX_OPT_LEVEL:
        return ()

    return (
        LINALG_SNITCH_OPTIMIZATION_PASSES[:-optimization_level]
        if optimization_level
        else LINALG_SNITCH_OPTIMIZATION_PASSES
    )


def get_passes(optimization_level: int = MAX_OPT_LEVEL) -> tuple[ModulePass, ...]:
    """
    This function returns a tuple of ModulePass objects to be applied in the lowering
    pipeline, based on the specified optimization level.

    Args:
        optimization_level (int): The desired optimization level, ranging from 0 to
            4 (inclusive). Defaults to 4.

    Returns:
        tuple[ModulePass, ...]: A tuple containing the ModulePass objects to be applied
        in the lowering pipeline.
    """

    excluded_passes = get_excluded_passes(optimization_level)
    return tuple(
        p for p in TEST_LOWER_LINALG_TO_SNITCH_PASSES if p not in excluded_passes
    )


@dataclass(frozen=True)
class TestLowerLinalgToSnitchPass(ModulePass):
    """
    A compiler pass used for testing lowering microkernels from linalg generic to snitch
    assembly.

    Args:
        optimization_level (int): The desired optimization level, ranging from 0 to
            4 (inclusive). Defaults to 4.
    """

    name = "test-lower-linalg-to-snitch"

    optimization_level: int = field(default=MAX_OPT_LEVEL)

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        passes = get_passes(self.optimization_level)

        for p in passes:
            p.apply(ctx, op)
