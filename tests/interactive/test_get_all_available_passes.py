from xdsl.backend.riscv.lowering import (
    convert_arith_to_riscv,
    convert_func_to_riscv_func,
)
from xdsl.interactive.get_all_available_passes import get_available_pass_list
from xdsl.interactive.passes import AvailablePass
from xdsl.interactive.rewrites import individual_rewrite
from xdsl.transforms import (
    mlir_opt,
    printf_to_llvm,
    reconcile_unrealized_casts,
    scf_parallel_loop_tiling,
    stencil_unroll,
)
from xdsl.transforms.experimental.dmp import stencil_global_to_local
from xdsl.utils.parse_pipeline import PipelinePassSpec


def test_get_all_available_passes():
    input_text = """
        func.func @hello(%n : index) -> index {
          %two = arith.constant 2 : index
          %res = arith.muli %two, %n : index
          func.return %res : index
        }
        """

    # Select two passes
    pass_pipeline = ()

    pass_pipeline = (
        *pass_pipeline,
        (
            convert_arith_to_riscv.ConvertArithToRiscvPass,
            PipelinePassSpec(name="convert-arith-to-riscv", args={}),
        ),
    )

    pass_pipeline = (
        *pass_pipeline,
        (
            convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass,
            PipelinePassSpec(name="convert-func-to-riscv-func", args={}),
        ),
    )

    expected_res = tuple(
        (
            AvailablePass(
                display_name="apply-individual-rewrite",
                module_pass=individual_rewrite.IndividualRewrite,
                pass_spec=None,
            ),
            AvailablePass(
                display_name="distribute-stencil",
                module_pass=stencil_global_to_local.DistributeStencilPass,
                pass_spec=None,
            ),
            AvailablePass(
                display_name="mlir-opt",
                module_pass=mlir_opt.MLIROptPass,
                pass_spec=None,
            ),
            AvailablePass(
                display_name="printf-to-llvm",
                module_pass=printf_to_llvm.PrintfToLLVM,
                pass_spec=None,
            ),
            AvailablePass(
                display_name="reconcile-unrealized-casts",
                module_pass=reconcile_unrealized_casts.ReconcileUnrealizedCastsPass,
                pass_spec=None,
            ),
            AvailablePass(
                display_name="scf-parallel-loop-tiling",
                module_pass=scf_parallel_loop_tiling.ScfParallelLoopTilingPass,
                pass_spec=None,
            ),
            AvailablePass(
                display_name="stencil-unroll",
                module_pass=stencil_unroll.StencilUnrollPass,
                pass_spec=None,
            ),
        )
    )

    res = get_available_pass_list(
        input_text,
        pass_pipeline,
        condense_mode=True,
        rewrite_by_names_dict=individual_rewrite.REWRITE_BY_NAMES,
    )

    assert res == expected_res
