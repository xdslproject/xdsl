from collections.abc import Callable

from xdsl.passes import ModulePass


def get_all_passes() -> dict[str, Callable[[], type[ModulePass]]]:
    """Return the list of all available passes."""

    def get_apply_individual_rewrite():
        from xdsl.transforms.individual_rewrite import ApplyIndividualRewritePass

        return ApplyIndividualRewritePass

    def get_apply_pdl():
        from xdsl.transforms import apply_pdl

        return apply_pdl.ApplyPDLPass

    def get_arith_add_fastmath():
        from xdsl.transforms import arith_add_fastmath

        return arith_add_fastmath.AddArithFastMathFlagsPass

    def get_canonicalize_dmp():
        from xdsl.transforms import canonicalize_dmp

        return canonicalize_dmp.CanonicalizeDmpPass

    def get_canonicalize():
        from xdsl.transforms import canonicalize

        return canonicalize.CanonicalizePass

    def get_constant_fold_interp():
        from xdsl.transforms import constant_fold_interp

        return constant_fold_interp.ConstantFoldInterpPass

    def get_control_flow_hoist():
        from xdsl.transforms import control_flow_hoist

        return control_flow_hoist.ControlFlowHoistPass

    def get_convert_arith_to_riscv_snitch():
        from xdsl.backend.riscv.lowering import convert_arith_to_riscv_snitch

        return convert_arith_to_riscv_snitch.ConvertArithToRiscvSnitchPass

    def get_convert_arith_to_riscv():
        from xdsl.backend.riscv.lowering import convert_arith_to_riscv

        return convert_arith_to_riscv.ConvertArithToRiscvPass

    def get_convert_arith_to_varith():
        from xdsl.transforms import varith_transformations

        return varith_transformations.ConvertArithToVarithPass

    def get_convert_func_to_riscv_func():
        from xdsl.backend.riscv.lowering import convert_func_to_riscv_func

        return convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass

    def get_convert_linalg_to_loops():
        from xdsl.transforms import convert_linalg_to_loops

        return convert_linalg_to_loops.ConvertLinalgToLoopsPass

    def get_convert_linalg_to_memref_stream():
        from xdsl.transforms import convert_linalg_to_memref_stream

        return convert_linalg_to_memref_stream.ConvertLinalgToMemrefStreamPass

    def get_convert_memref_stream_to_loops():
        from xdsl.transforms import convert_memref_stream_to_loops

        return convert_memref_stream_to_loops.ConvertMemrefStreamToLoopsPass

    def get_convert_memref_stream_to_snitch_stream():
        from xdsl.transforms import convert_memref_stream_to_snitch_stream

        return (
            convert_memref_stream_to_snitch_stream.ConvertMemrefStreamToSnitchStreamPass
        )

    def get_convert_memref_to_ptr():
        from xdsl.transforms import convert_memref_to_ptr

        return convert_memref_to_ptr.ConvertMemrefToPtr

    def get_convert_memref_to_riscv():
        from xdsl.backend.riscv.lowering import convert_memref_to_riscv

        return convert_memref_to_riscv.ConvertMemrefToRiscvPass

    def get_convert_ml_program_to_memref():
        from xdsl.transforms import convert_ml_program_to_memref

        return convert_ml_program_to_memref.ConvertMlProgramToMemrefPass

    def get_convert_print_format_to_riscv_debug():
        from xdsl.backend.riscv.lowering import convert_print_format_to_riscv_debug

        return convert_print_format_to_riscv_debug.ConvertPrintFormatToRiscvDebugPass

    def get_convert_ptr_to_riscv():
        from xdsl.transforms import convert_ptr_to_riscv

        return convert_ptr_to_riscv.ConvertPtrToRiscvPass

    def get_convert_qref_to_qssa():
        from xdsl.transforms import convert_qref_to_qssa

        return convert_qref_to_qssa.ConvertQRefToQssa

    def get_convert_qssa_to_qref():
        from xdsl.transforms import convert_qssa_to_qref

        return convert_qssa_to_qref.ConvertQssaToQRef

    def get_convert_riscv_scf_for_to_frep():
        from xdsl.transforms import convert_riscv_scf_for_to_frep

        return convert_riscv_scf_for_to_frep.ConvertRiscvScfForToFrepPass

    def get_convert_riscv_scf_to_riscv_cf():
        from xdsl.backend.riscv.lowering import convert_riscv_scf_to_riscv_cf

        return convert_riscv_scf_to_riscv_cf.ConvertRiscvScfToRiscvCfPass

    def get_convert_riscv_to_llvm():
        from xdsl.transforms import convert_riscv_to_llvm

        return convert_riscv_to_llvm.ConvertRiscvToLLVMPass

    def get_convert_scf_to_cf():
        from xdsl.transforms import convert_scf_to_cf

        return convert_scf_to_cf.ConvertScfToCf

    def get_convert_scf_to_openmp():
        from xdsl.transforms import convert_scf_to_openmp

        return convert_scf_to_openmp.ConvertScfToOpenMPPass

    def get_convert_scf_to_riscv_scf():
        from xdsl.backend.riscv.lowering import convert_scf_to_riscv_scf

        return convert_scf_to_riscv_scf.ConvertScfToRiscvPass

    def get_convert_snitch_stream_to_snitch():
        from xdsl.backend.riscv.lowering import convert_snitch_stream_to_snitch

        return convert_snitch_stream_to_snitch.ConvertSnitchStreamToSnitch

    def get_convert_stencil_to_csl_stencil():
        from xdsl.transforms import convert_stencil_to_csl_stencil

        return convert_stencil_to_csl_stencil.ConvertStencilToCslStencilPass

    def get_convert_stencil_to_ll_mlir():
        from xdsl.transforms.experimental import convert_stencil_to_ll_mlir

        return convert_stencil_to_ll_mlir.ConvertStencilToLLMLIRPass

    def get_convert_varith_to_arith():
        from xdsl.transforms import varith_transformations

        return varith_transformations.ConvertVarithToArithPass

    def get_jax_use_donated_arguments():
        from xdsl.transforms import jax_use_donated_arguments

        return jax_use_donated_arguments.JaxUseDonatedArguments

    def get_cse():
        from xdsl.transforms import common_subexpression_elimination

        return common_subexpression_elimination.CommonSubexpressionElimination

    def get_csl_stencil_bufferize():
        from xdsl.transforms import csl_stencil_bufferize

        return csl_stencil_bufferize.CslStencilBufferize

    def get_csl_stencil_handle_async_flow():
        from xdsl.transforms import csl_stencil_handle_async_flow

        return csl_stencil_handle_async_flow.CslStencilHandleAsyncControlFlow

    def get_csl_stencil_materialize_stores():
        from xdsl.transforms import csl_stencil_materialize_stores

        return csl_stencil_materialize_stores.CslStencilMaterializeStores

    def get_csl_stencil_to_csl_wrapper():
        from xdsl.transforms import csl_stencil_to_csl_wrapper

        return csl_stencil_to_csl_wrapper.CslStencilToCslWrapperPass

    def get_csl_wrapper_hoist_buffers():
        from xdsl.transforms import csl_wrapper_hoist_buffers

        return csl_wrapper_hoist_buffers.CslWrapperHoistBuffers

    def get_dce():
        from xdsl.transforms import dead_code_elimination

        return dead_code_elimination.DeadCodeElimination

    def get_distribute_stencil():
        from xdsl.transforms.experimental.dmp import stencil_global_to_local

        return stencil_global_to_local.DistributeStencilPass

    def get_dmp_to_mpi():
        from xdsl.transforms.experimental.dmp import stencil_global_to_local

        return stencil_global_to_local.DmpToMpiPass

    def get_empty_tensor_to_alloc_tensor():
        from xdsl.transforms import empty_tensor_to_alloc_tensor

        return empty_tensor_to_alloc_tensor.EmptyTensorToAllocTensorPass

    def get_eqsat_add_costs():
        from xdsl.transforms import eqsat_add_costs

        return eqsat_add_costs.EqsatAddCostsPass

    def get_eqsat_create_eclasses():
        from xdsl.transforms import eqsat_create_eclasses

        return eqsat_create_eclasses.EqsatCreateEclassesPass

    def get_eqsat_extract():
        from xdsl.transforms import eqsat_extract

        return eqsat_extract.EqsatExtractPass

    def get_frontend_desymrefy():
        from xdsl.frontend.passes.desymref import FrontendDesymrefyPass

        return FrontendDesymrefyPass

    def get_function_constant_pinning():
        from xdsl.transforms.experimental import function_constant_pinning

        return function_constant_pinning.FunctionConstantPinningPass

    def get_function_persist_arg_names():
        from xdsl.transforms import function_transformations

        return function_transformations.FunctionPersistArgNamesPass

    def get_gpu_map_parallel_loops():
        from xdsl.transforms import gpu_map_parallel_loops

        return gpu_map_parallel_loops.GpuMapParallelLoopsPass

    def get_hls_convert_stencil_to_ll_mlir():
        from xdsl.transforms.experimental import hls_convert_stencil_to_ll_mlir

        return hls_convert_stencil_to_ll_mlir.HLSConvertStencilToLLMLIRPass

    def get_inline_snrt():
        from xdsl.transforms import inline_snrt

        return inline_snrt.InlineSnrtPass

    def get_lift_arith_to_linalg():
        from xdsl.transforms.lift_arith_to_linalg import LiftArithToLinalg

        return LiftArithToLinalg

    def get_linalg_fuse_multiply_add():
        from xdsl.transforms.linalg_transformations import LinalgFuseMultiplyAddPass

        return LinalgFuseMultiplyAddPass

    def get_linalg_to_csl():
        from xdsl.transforms.linalg_to_csl import LinalgToCsl

        return LinalgToCsl

    def get_loop_hoist_memref():
        from xdsl.transforms import loop_hoist_memref

        return loop_hoist_memref.LoopHoistMemrefPass

    def get_lower_affine():
        from xdsl.transforms import lower_affine

        return lower_affine.LowerAffinePass

    def get_lower_csl_stencil():
        from xdsl.transforms import lower_csl_stencil

        return lower_csl_stencil.LowerCslStencil

    def get_lower_csl_wrapper():
        from xdsl.transforms import lower_csl_wrapper

        return lower_csl_wrapper.LowerCslWrapperPass

    def get_lower_hls():
        from xdsl.transforms.experimental import lower_hls

        return lower_hls.LowerHLSPass

    def get_lower_mpi():
        from xdsl.transforms import lower_mpi

        return lower_mpi.LowerMPIPass

    def get_lower_riscv_func():
        from xdsl.transforms import lower_riscv_func

        return lower_riscv_func.LowerRISCVFunc

    def get_lower_riscv_scf_to_labels():
        from xdsl.backend.riscv import riscv_scf_to_asm

        return riscv_scf_to_asm.LowerRiscvScfForToLabelsPass

    def get_lower_snitch():
        from xdsl.transforms import lower_snitch

        return lower_snitch.LowerSnitchPass

    def get_memref_stream_fold_fill():
        from xdsl.transforms import memref_stream_fold_fill

        return memref_stream_fold_fill.MemrefStreamFoldFillPass

    def get_memref_stream_generalize_fill():
        from xdsl.transforms import memref_stream_generalize_fill

        return memref_stream_generalize_fill.MemrefStreamGeneralizeFillPass

    def get_memref_stream_infer_fill():
        from xdsl.transforms import memref_stream_infer_fill

        return memref_stream_infer_fill.MemrefStreamInferFillPass

    def get_memref_stream_interleave():
        from xdsl.transforms import memref_stream_interleave

        return memref_stream_interleave.MemrefStreamInterleavePass

    def get_memref_stream_legalize():
        from xdsl.transforms import memref_stream_legalize

        return memref_stream_legalize.MemrefStreamLegalizePass

    def get_memref_stream_tile_outer_loops():
        from xdsl.transforms import memref_stream_tile_outer_loops

        return memref_stream_tile_outer_loops.MemrefStreamTileOuterLoopsPass

    def get_memref_stream_unnest_out_parameters():
        from xdsl.transforms import memref_stream_unnest_out_parameters

        return memref_stream_unnest_out_parameters.MemrefStreamUnnestOutParametersPass

    def get_memref_streamify():
        from xdsl.transforms import memref_streamify

        return memref_streamify.MemrefStreamifyPass

    def get_memref_to_dsd():
        from xdsl.transforms import memref_to_dsd

        return memref_to_dsd.MemrefToDsdPass

    def get_memref_to_gpu():
        from xdsl.transforms import gpu_allocs

        return gpu_allocs.MemrefToGPUPass

    def get_mlir_opt():
        from xdsl.transforms import mlir_opt

        return mlir_opt.MLIROptPass

    def get_printf_to_llvm():
        from xdsl.transforms import printf_to_llvm

        return printf_to_llvm.PrintfToLLVM

    def get_printf_to_putchar():
        from xdsl.transforms import printf_to_putchar

        return printf_to_putchar.PrintfToPutcharPass

    def get_reconcile_unrealized_casts():
        from xdsl.transforms import reconcile_unrealized_casts

        return reconcile_unrealized_casts.ReconcileUnrealizedCastsPass

    def get_replace_incompatible_fpga():
        from xdsl.transforms.experimental import replace_incompatible_fpga

        return replace_incompatible_fpga.ReplaceIncompatibleFPGA

    def get_riscv_register_allocation():
        from xdsl.transforms import riscv_register_allocation

        return riscv_register_allocation.RISCVRegisterAllocation

    def get_riscv_prologue_epilogue_insertion():
        from xdsl.backend.riscv import prologue_epilogue_insertion

        return prologue_epilogue_insertion.PrologueEpilogueInsertion

    def get_riscv_scf_loop_range_folding():
        from xdsl.transforms import riscv_scf_loop_range_folding

        return riscv_scf_loop_range_folding.RiscvScfLoopRangeFoldingPass

    def get_scf_for_loop_flatten():
        from xdsl.transforms import scf_for_loop_flatten

        return scf_for_loop_flatten.ScfForLoopFlattenPass

    def get_scf_for_loop_range_folding():
        from xdsl.transforms import scf_for_loop_range_folding

        return scf_for_loop_range_folding.ScfForLoopRangeFoldingPass

    def get_scf_parallel_loop_tiling():
        from xdsl.transforms import scf_parallel_loop_tiling

        return scf_parallel_loop_tiling.ScfParallelLoopTilingPass

    def get_shape_inference():
        from xdsl.transforms.shape_inference import ShapeInferencePass

        return ShapeInferencePass

    def get_snitch_allocate_registers():
        from xdsl.transforms import snitch_register_allocation

        return snitch_register_allocation.SnitchRegisterAllocation

    def get_stencil_bufferize():
        from xdsl.transforms import stencil_bufferize

        return stencil_bufferize.StencilBufferize

    def get_stencil_inlining():
        from xdsl.transforms import stencil_inlining

        return stencil_inlining.StencilInliningPass

    def get_stencil_shape_minimize():
        from xdsl.transforms import stencil_shape_minimize

        return stencil_shape_minimize.StencilShapeMinimize

    def get_stencil_storage_materialization():
        from xdsl.transforms.experimental import stencil_storage_materialization

        return stencil_storage_materialization.StencilStorageMaterializationPass

    def get_stencil_tensorize_z_dimension():
        from xdsl.transforms.experimental import stencil_tensorize_z_dimension

        return stencil_tensorize_z_dimension.StencilTensorizeZDimension

    def get_stencil_unroll():
        from xdsl.transforms import stencil_unroll

        return stencil_unroll.StencilUnrollPass

    def get_test_add_timers_to_top_level_funcs():
        from xdsl.transforms import function_transformations

        return function_transformations.TestAddBenchTimersToTopLevelFunctions

    def get_test_lower_linalg_to_snitch():
        from xdsl.transforms import test_lower_linalg_to_snitch

        return test_lower_linalg_to_snitch.TestLowerLinalgToSnitchPass

    def get_varith_fuse_repeated_operands():
        from xdsl.transforms import varith_transformations

        return varith_transformations.VarithFuseRepeatedOperandsPass

    # Please insert pass and `get_` function in alphabetical order

    return {
        "apply-individual-rewrite": get_apply_individual_rewrite,
        "apply-pdl": get_apply_pdl,
        "arith-add-fastmath": get_arith_add_fastmath,
        "canonicalize-dmp": get_canonicalize_dmp,
        "canonicalize": get_canonicalize,
        "constant-fold-interp": get_constant_fold_interp,
        "control-flow-hoist": get_control_flow_hoist,
        "convert-arith-to-riscv-snitch": get_convert_arith_to_riscv_snitch,
        "convert-arith-to-riscv": get_convert_arith_to_riscv,
        "convert-arith-to-varith": get_convert_arith_to_varith,
        "convert-func-to-riscv-func": get_convert_func_to_riscv_func,
        "convert-linalg-to-loops": get_convert_linalg_to_loops,
        "convert-linalg-to-memref-stream": get_convert_linalg_to_memref_stream,
        "convert-memref-stream-to-loops": get_convert_memref_stream_to_loops,
        "convert-memref-stream-to-snitch-stream": get_convert_memref_stream_to_snitch_stream,
        "convert-memref-to-ptr": get_convert_memref_to_ptr,
        "convert-memref-to-riscv": get_convert_memref_to_riscv,
        "convert-ml-program-to-memref": get_convert_ml_program_to_memref,
        "convert-print-format-to-riscv-debug": get_convert_print_format_to_riscv_debug,
        "convert-ptr-to-riscv": get_convert_ptr_to_riscv,
        "convert-qref-to-qssa": get_convert_qref_to_qssa,
        "convert-qssa-to-qref": get_convert_qssa_to_qref,
        "convert-riscv-scf-for-to-frep": get_convert_riscv_scf_for_to_frep,
        "convert-riscv-scf-to-riscv-cf": get_convert_riscv_scf_to_riscv_cf,
        "convert-riscv-to-llvm": get_convert_riscv_to_llvm,
        "convert-scf-to-cf": get_convert_scf_to_cf,
        "convert-scf-to-openmp": get_convert_scf_to_openmp,
        "convert-scf-to-riscv-scf": get_convert_scf_to_riscv_scf,
        "convert-snitch-stream-to-snitch": get_convert_snitch_stream_to_snitch,
        "convert-stencil-to-csl-stencil": get_convert_stencil_to_csl_stencil,
        "convert-stencil-to-ll-mlir": get_convert_stencil_to_ll_mlir,
        "convert-varith-to-arith": get_convert_varith_to_arith,
        "jax-use-donated-arguments": get_jax_use_donated_arguments,
        "cse": get_cse,
        "csl-stencil-bufferize": get_csl_stencil_bufferize,
        "csl-stencil-handle-async-flow": get_csl_stencil_handle_async_flow,
        "csl-stencil-materialize-stores": get_csl_stencil_materialize_stores,
        "csl-stencil-to-csl-wrapper": get_csl_stencil_to_csl_wrapper,
        "csl-wrapper-hoist-buffers": get_csl_wrapper_hoist_buffers,
        "dce": get_dce,
        "distribute-stencil": get_distribute_stencil,
        "dmp-to-mpi": get_dmp_to_mpi,
        "empty-tensor-to-alloc-tensor": get_empty_tensor_to_alloc_tensor,
        "eqsat-add-costs": get_eqsat_add_costs,
        "eqsat-create-eclasses": get_eqsat_create_eclasses,
        "eqsat-extract": get_eqsat_extract,
        "frontend-desymrefy": get_frontend_desymrefy,
        "function-constant-pinning": get_function_constant_pinning,
        "function-persist-arg-names": get_function_persist_arg_names,
        "gpu-map-parallel-loops": get_gpu_map_parallel_loops,
        "hls-convert-stencil-to-ll-mlir": get_hls_convert_stencil_to_ll_mlir,
        "inline-snrt": get_inline_snrt,
        "lift-arith-to-linalg": get_lift_arith_to_linalg,
        "linalg-fuse-multiply-add": get_linalg_fuse_multiply_add,
        "linalg-to-csl": get_linalg_to_csl,
        "loop-hoist-memref": get_loop_hoist_memref,
        "lower-affine": get_lower_affine,
        "lower-csl-stencil": get_lower_csl_stencil,
        "lower-csl-wrapper": get_lower_csl_wrapper,
        "lower-hls": get_lower_hls,
        "lower-mpi": get_lower_mpi,
        "lower-riscv-func": get_lower_riscv_func,
        "lower-riscv-scf-to-labels": get_lower_riscv_scf_to_labels,
        "lower-snitch": get_lower_snitch,
        "memref-stream-fold-fill": get_memref_stream_fold_fill,
        "memref-stream-generalize-fill": get_memref_stream_generalize_fill,
        "memref-stream-infer-fill": get_memref_stream_infer_fill,
        "memref-stream-interleave": get_memref_stream_interleave,
        "memref-stream-legalize": get_memref_stream_legalize,
        "memref-stream-tile-outer-loops": get_memref_stream_tile_outer_loops,
        "memref-stream-unnest-out-parameters": get_memref_stream_unnest_out_parameters,
        "memref-streamify": get_memref_streamify,
        "memref-to-dsd": get_memref_to_dsd,
        "memref-to-gpu": get_memref_to_gpu,
        "mlir-opt": get_mlir_opt,
        "printf-to-llvm": get_printf_to_llvm,
        "printf-to-putchar": get_printf_to_putchar,
        "reconcile-unrealized-casts": get_reconcile_unrealized_casts,
        "replace-incompatible-fpga": get_replace_incompatible_fpga,
        "riscv-allocate-registers": get_riscv_register_allocation,
        "riscv-prologue-epilogue-insertion": get_riscv_prologue_epilogue_insertion,
        "riscv-scf-loop-range-folding": get_riscv_scf_loop_range_folding,
        "scf-for-loop-flatten": get_scf_for_loop_flatten,
        "scf-for-loop-range-folding": get_scf_for_loop_range_folding,
        "scf-parallel-loop-tiling": get_scf_parallel_loop_tiling,
        "shape-inference": get_shape_inference,
        "snitch-allocate-registers": get_snitch_allocate_registers,
        "stencil-bufferize": get_stencil_bufferize,
        "stencil-inlining": get_stencil_inlining,
        "stencil-shape-minimize": get_stencil_shape_minimize,
        "stencil-storage-materialization": get_stencil_storage_materialization,
        "stencil-tensorize-z-dimension": get_stencil_tensorize_z_dimension,
        "stencil-unroll": get_stencil_unroll,
        "test-add-timers-to-top-level-funcs": get_test_add_timers_to_top_level_funcs,
        "test-lower-linalg-to-snitch": get_test_lower_linalg_to_snitch,
        "varith-fuse-repeated-operands": get_varith_fuse_repeated_operands,
    }
