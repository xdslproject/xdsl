from collections.abc import Callable

from xdsl.passes import ModulePass


def get_all_passes() -> dict[str, Callable[[], type[ModulePass]]]:
    """Return the list of all available passes."""

    def get_apply_individual_rewrite():
        from xdsl.transforms.individual_rewrite import ApplyIndividualRewritePass

        return ApplyIndividualRewritePass

    def get_apply_eqsat_pdl():
        from xdsl.transforms import apply_eqsat_pdl

        return apply_eqsat_pdl.ApplyEqsatPDLPass

    def get_apply_eqsat_pdl_interp():
        from xdsl.transforms import apply_eqsat_pdl_interp

        return apply_eqsat_pdl_interp.ApplyEqsatPDLInterpPass

    def get_apply_pdl():
        from xdsl.transforms import apply_pdl

        return apply_pdl.ApplyPDLPass

    def get_apply_pdl_interp():
        from xdsl.transforms import apply_pdl_interp

        return apply_pdl_interp.ApplyPDLInterpPass

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

    def get_convert_arith_to_x86():
        from xdsl.backend.x86.lowering import convert_arith_to_x86

        return convert_arith_to_x86.ConvertArithToX86Pass

    def get_convert_func_to_riscv_func():
        from xdsl.backend.riscv.lowering import convert_func_to_riscv_func

        return convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass

    def get_convert_func_to_x86_func():
        from xdsl.backend.x86.lowering import convert_func_to_x86_func

        return convert_func_to_x86_func.ConvertFuncToX86FuncPass

    def get_convert_linalg_to_loops():
        from xdsl.transforms import convert_linalg_to_loops

        return convert_linalg_to_loops.ConvertLinalgToLoopsPass

    def get_convert_linalg_to_memref_stream():
        from xdsl.transforms import convert_linalg_to_memref_stream

        return convert_linalg_to_memref_stream.ConvertLinalgToMemRefStreamPass

    def get_convert_memref_stream_to_loops():
        from xdsl.transforms import convert_memref_stream_to_loops

        return convert_memref_stream_to_loops.ConvertMemRefStreamToLoopsPass

    def get_convert_memref_stream_to_snitch_stream():
        from xdsl.transforms import convert_memref_stream_to_snitch_stream

        return (
            convert_memref_stream_to_snitch_stream.ConvertMemRefStreamToSnitchStreamPass
        )

    def get_convert_memref_to_ptr():
        from xdsl.transforms import convert_memref_to_ptr

        return convert_memref_to_ptr.ConvertMemRefToPtr

    def get_convert_memref_to_riscv():
        from xdsl.backend.riscv.lowering import convert_memref_to_riscv

        return convert_memref_to_riscv.ConvertMemRefToRiscvPass

    def get_convert_ml_program_to_memref():
        from xdsl.transforms import convert_ml_program_to_memref

        return convert_ml_program_to_memref.ConvertMlProgramToMemRefPass

    def get_convert_print_format_to_riscv_debug():
        from xdsl.backend.riscv.lowering import convert_print_format_to_riscv_debug

        return convert_print_format_to_riscv_debug.ConvertPrintFormatToRiscvDebugPass

    def get_convert_ptr_to_llvm():
        from xdsl.transforms import convert_ptr_to_llvm

        return convert_ptr_to_llvm.ConvertPtrToLLVMPass

    def get_convert_ptr_to_riscv():
        from xdsl.transforms import convert_ptr_to_riscv

        return convert_ptr_to_riscv.ConvertPtrToRiscvPass

    def get_convert_ptr_to_x86():
        from xdsl.backend.x86.lowering import convert_ptr_to_x86

        return convert_ptr_to_x86.ConvertPtrToX86Pass

    def get_convert_ptr_type_offsets():
        from xdsl.transforms import convert_ptr_type_offsets

        return convert_ptr_type_offsets.ConvertPtrTypeOffsetsPass

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

    def get_convert_scf_to_x86_scf():
        from xdsl.transforms import convert_scf_to_x86_scf

        return convert_scf_to_x86_scf.ConvertScfToX86ScfPass

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

    def get_convert_vector_to_ptr():
        from xdsl.transforms import convert_vector_to_ptr

        return convert_vector_to_ptr.ConvertVectorToPtrPass

    def get_convert_vector_to_x86():
        from xdsl.backend.x86.lowering import convert_vector_to_x86

        return convert_vector_to_x86.ConvertVectorToX86Pass

    def get_convert_x86_scf_to_x86():
        from xdsl.transforms import convert_x86_scf_to_x86

        return convert_x86_scf_to_x86.ConvertX86ScfToX86Pass

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

    def get_csl_stencil_set_global_coeffs():
        from xdsl.transforms import csl_stencil_set_global_coeffs

        return csl_stencil_set_global_coeffs.CslStencilSetGlobalCoeffs

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

    def get_eqsat_create_egraphs():
        from xdsl.transforms import eqsat_create_egraphs

        return eqsat_create_egraphs.EqsatCreateEgraphsPass

    def get_eqsat_serialize_egraph():
        from xdsl.transforms import eqsat_serialize_egraph

        return eqsat_serialize_egraph.SerializeEGraph

    def get_eqsat_extract():
        from xdsl.transforms import eqsat_extract

        return eqsat_extract.EqsatExtractPass

    def get_frontend_desymrefy():
        from xdsl.transforms.desymref import FrontendDesymrefyPass

        return FrontendDesymrefyPass

    def get_function_constant_pinning():
        from xdsl.transforms.experimental import function_constant_pinning

        return function_constant_pinning.FunctionConstantPinningPass

    def get_function_persist_arg_names():
        from xdsl.transforms import function_transformations

        return function_transformations.FunctionPersistArgNamesPass

    def get_func_to_pdl_rewrite():
        from xdsl.transforms.experimental import func_to_pdl_rewrite

        return func_to_pdl_rewrite.FuncToPdlRewrite

    def get_gpu_map_parallel_loops():
        from xdsl.transforms import gpu_map_parallel_loops

        return gpu_map_parallel_loops.GpuMapParallelLoopsPass

    def get_hls_convert_stencil_to_ll_mlir():
        from xdsl.transforms.experimental import hls_convert_stencil_to_ll_mlir

        return hls_convert_stencil_to_ll_mlir.HLSConvertStencilToLLMLIRPass

    def get_inline_snrt():
        from xdsl.transforms import inline_snrt

        return inline_snrt.InlineSnrtPass

    def get_licm():
        from xdsl.transforms import loop_invariant_code_motion

        return loop_invariant_code_motion.LoopInvariantCodeMotionPass

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

        return loop_hoist_memref.LoopHoistMemRefPass

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

        return memref_stream_fold_fill.MemRefStreamFoldFillPass

    def get_memref_stream_generalize_fill():
        from xdsl.transforms import memref_stream_generalize_fill

        return memref_stream_generalize_fill.MemRefStreamGeneralizeFillPass

    def get_memref_stream_infer_fill():
        from xdsl.transforms import memref_stream_infer_fill

        return memref_stream_infer_fill.MemRefStreamInferFillPass

    def get_memref_stream_interleave():
        from xdsl.transforms import memref_stream_interleave

        return memref_stream_interleave.MemRefStreamInterleavePass

    def get_memref_stream_legalize():
        from xdsl.transforms import memref_stream_legalize

        return memref_stream_legalize.MemRefStreamLegalizePass

    def get_memref_stream_tile_outer_loops():
        from xdsl.transforms import memref_stream_tile_outer_loops

        return memref_stream_tile_outer_loops.MemRefStreamTileOuterLoopsPass

    def get_memref_stream_unnest_out_parameters():
        from xdsl.transforms import memref_stream_unnest_out_parameters

        return memref_stream_unnest_out_parameters.MemRefStreamUnnestOutParametersPass

    def get_memref_streamify():
        from xdsl.transforms import memref_streamify

        return memref_streamify.MemRefStreamifyPass

    def get_memref_to_dsd():
        from xdsl.transforms import memref_to_dsd

        return memref_to_dsd.MemRefToDsdPass

    def get_memref_to_gpu():
        from xdsl.transforms import gpu_allocs

        return gpu_allocs.MemRefToGPUPass

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

    def get_riscv_allocate_registers():
        from xdsl.transforms import riscv_allocate_registers

        return riscv_allocate_registers.RISCVAllocateRegistersPass

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

    def get_scf_for_loop_unroll():
        from xdsl.transforms import scf_for_loop_unroll

        return scf_for_loop_unroll.ScfForLoopUnrollPass

    def get_scf_parallel_loop_tiling():
        from xdsl.transforms import scf_parallel_loop_tiling

        return scf_parallel_loop_tiling.ScfParallelLoopTilingPass

    def get_shape_inference():
        from xdsl.transforms.shape_inference import ShapeInferencePass

        return ShapeInferencePass

    def get_snitch_allocate_registers():
        from xdsl.transforms import snitch_allocate_registers

        return snitch_allocate_registers.SnitchAllocateRegistersPass

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

    def get_test_constant_folding():
        from xdsl.transforms import test_constant_folding

        return test_constant_folding.TestConstantFoldingPass

    def get_test_lower_linalg_to_snitch():
        from xdsl.transforms import test_lower_linalg_to_snitch

        return test_lower_linalg_to_snitch.TestLowerLinalgToSnitchPass

    def get_test_specialised_constant_folding():
        from xdsl.transforms import test_constant_folding

        return test_constant_folding.TestSpecialisedConstantFoldingPass

    def get_test_transform_dialect_erase_schedule():
        from xdsl.transforms import test_transform_dialect_erase_schedule

        return (
            test_transform_dialect_erase_schedule.TestTransformDialectEraseSchedulePass
        )

    def get_test_vectorize_matmul():
        from xdsl.transforms import test_vectorize_matmul

        return test_vectorize_matmul.TestVectorizeMatmulPass

    def get_transform_interpreter():
        from xdsl.transforms import transform_interpreter

        return transform_interpreter.TransformInterpreterPass

    def get_varith_fuse_repeated_operands():
        from xdsl.transforms import varith_transformations

        return varith_transformations.VarithFuseRepeatedOperandsPass

    def get_vector_split_load_extract():
        from xdsl.transforms import vector_split_load_extract

        return vector_split_load_extract.VectorSplitLoadExtractPass

    def get_x86_allocate_registers():
        from xdsl.transforms import x86_allocate_registers

        return x86_allocate_registers.X86AllocateRegisters

    def get_x86_prologue_epilogue_insertion():
        from xdsl.backend.x86 import prologue_epilogue_insertion

        return prologue_epilogue_insertion.X86PrologueEpilogueInsertion

    def get_x86_infer_broadcast():
        from xdsl.transforms import x86_infer_broadcast

        return x86_infer_broadcast.X86InferBroadcast

    def get_verify_register_allocation():
        from xdsl.transforms import verify_register_allocation

        return verify_register_allocation.VerifyRegisterAllocationPass

    # Please insert pass and `get_` function in alphabetical order

    return {
        "apply-individual-rewrite": get_apply_individual_rewrite,
        "apply-eqsat-pdl": get_apply_eqsat_pdl,
        "apply-eqsat-pdl-interp": get_apply_eqsat_pdl_interp,
        "apply-pdl": get_apply_pdl,
        "apply-pdl-interp": get_apply_pdl_interp,
        "arith-add-fastmath": get_arith_add_fastmath,
        "canonicalize-dmp": get_canonicalize_dmp,
        "canonicalize": get_canonicalize,
        "constant-fold-interp": get_constant_fold_interp,
        "control-flow-hoist": get_control_flow_hoist,
        "convert-arith-to-riscv-snitch": get_convert_arith_to_riscv_snitch,
        "convert-arith-to-riscv": get_convert_arith_to_riscv,
        "convert-arith-to-varith": get_convert_arith_to_varith,
        "convert-arith-to-x86": get_convert_arith_to_x86,
        "convert-func-to-riscv-func": get_convert_func_to_riscv_func,
        "convert-func-to-x86-func": get_convert_func_to_x86_func,
        "convert-linalg-to-loops": get_convert_linalg_to_loops,
        "convert-linalg-to-memref-stream": get_convert_linalg_to_memref_stream,
        "convert-memref-stream-to-loops": get_convert_memref_stream_to_loops,
        "convert-memref-stream-to-snitch-stream": get_convert_memref_stream_to_snitch_stream,
        "convert-memref-to-ptr": get_convert_memref_to_ptr,
        "convert-memref-to-riscv": get_convert_memref_to_riscv,
        "convert-ml-program-to-memref": get_convert_ml_program_to_memref,
        "convert-print-format-to-riscv-debug": get_convert_print_format_to_riscv_debug,
        "convert-ptr-to-llvm": get_convert_ptr_to_llvm,
        "convert-ptr-to-riscv": get_convert_ptr_to_riscv,
        "convert-ptr-to-x86": get_convert_ptr_to_x86,
        "convert-riscv-scf-for-to-frep": get_convert_riscv_scf_for_to_frep,
        "convert-riscv-scf-to-riscv-cf": get_convert_riscv_scf_to_riscv_cf,
        "convert-riscv-to-llvm": get_convert_riscv_to_llvm,
        "convert-scf-to-cf": get_convert_scf_to_cf,
        "convert-scf-to-openmp": get_convert_scf_to_openmp,
        "convert-scf-to-riscv-scf": get_convert_scf_to_riscv_scf,
        "convert-scf-to-x86-scf": get_convert_scf_to_x86_scf,
        "convert-snitch-stream-to-snitch": get_convert_snitch_stream_to_snitch,
        "convert-ptr-type-offsets": get_convert_ptr_type_offsets,
        "convert-stencil-to-csl-stencil": get_convert_stencil_to_csl_stencil,
        "convert-stencil-to-ll-mlir": get_convert_stencil_to_ll_mlir,
        "convert-varith-to-arith": get_convert_varith_to_arith,
        "convert-vector-to-ptr": get_convert_vector_to_ptr,
        "convert-vector-to-x86": get_convert_vector_to_x86,
        "convert-x86-scf-to-x86": get_convert_x86_scf_to_x86,
        "jax-use-donated-arguments": get_jax_use_donated_arguments,
        "cse": get_cse,
        "csl-stencil-bufferize": get_csl_stencil_bufferize,
        "csl-stencil-handle-async-flow": get_csl_stencil_handle_async_flow,
        "csl-stencil-materialize-stores": get_csl_stencil_materialize_stores,
        "csl-stencil-set-global-coeffs": get_csl_stencil_set_global_coeffs,
        "csl-stencil-to-csl-wrapper": get_csl_stencil_to_csl_wrapper,
        "csl-wrapper-hoist-buffers": get_csl_wrapper_hoist_buffers,
        "dce": get_dce,
        "distribute-stencil": get_distribute_stencil,
        "dmp-to-mpi": get_dmp_to_mpi,
        "empty-tensor-to-alloc-tensor": get_empty_tensor_to_alloc_tensor,
        "eqsat-add-costs": get_eqsat_add_costs,
        "eqsat-create-eclasses": get_eqsat_create_eclasses,
        "eqsat-create-egraphs": get_eqsat_create_egraphs,
        "eqsat-serialize-egraph": get_eqsat_serialize_egraph,
        "eqsat-extract": get_eqsat_extract,
        "frontend-desymrefy": get_frontend_desymrefy,
        "function-constant-pinning": get_function_constant_pinning,
        "function-persist-arg-names": get_function_persist_arg_names,
        "func-to-pdl-rewrite": get_func_to_pdl_rewrite,
        "gpu-map-parallel-loops": get_gpu_map_parallel_loops,
        "hls-convert-stencil-to-ll-mlir": get_hls_convert_stencil_to_ll_mlir,
        "inline-snrt": get_inline_snrt,
        "licm": get_licm,
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
        "riscv-allocate-registers": get_riscv_allocate_registers,
        "riscv-prologue-epilogue-insertion": get_riscv_prologue_epilogue_insertion,
        "riscv-scf-loop-range-folding": get_riscv_scf_loop_range_folding,
        "scf-for-loop-flatten": get_scf_for_loop_flatten,
        "scf-for-loop-range-folding": get_scf_for_loop_range_folding,
        "scf-for-loop-unroll": get_scf_for_loop_unroll,
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
        "test-constant-folding": get_test_constant_folding,
        "test-lower-linalg-to-snitch": get_test_lower_linalg_to_snitch,
        "test-specialised-constant-folding": get_test_specialised_constant_folding,
        "test-transform-dialect-erase-schedule": get_test_transform_dialect_erase_schedule,
        "test-vectorize-matmul": get_test_vectorize_matmul,
        "transform-interpreter": get_transform_interpreter,
        "varith-fuse-repeated-operands": get_varith_fuse_repeated_operands,
        "vector-split-load-extract": get_vector_split_load_extract,
        "x86-allocate-registers": get_x86_allocate_registers,
        "x86-prologue-epilogue-insertion": get_x86_prologue_epilogue_insertion,
        "x86-infer-broadcast": get_x86_infer_broadcast,
        "verify-register-allocation": get_verify_register_allocation,
    }
