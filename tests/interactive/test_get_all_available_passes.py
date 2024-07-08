from xdsl.backend.riscv.lowering import (
    convert_arith_to_riscv,
    convert_func_to_riscv_func,
)
from xdsl.interactive.get_all_available_passes import get_available_pass_list
from xdsl.interactive.passes import AvailablePass
from xdsl.interactive.rewrites import individual_rewrite
from xdsl.transforms import (
    reconcile_unrealized_casts,
    test_lower_memref_stream_to_snitch_stream,
)
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
    pass_pipeline = (
        (
            convert_arith_to_riscv.ConvertArithToRiscvPass,
            PipelinePassSpec(name="convert-arith-to-riscv", args={}),
        ),
        (
            convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass,
            PipelinePassSpec(name="convert-func-to-riscv-func", args={}),
        ),
    )

    expected_res = tuple(
        (
            AvailablePass(
                display_name="reconcile-unrealized-casts",
                module_pass=reconcile_unrealized_casts.ReconcileUnrealizedCastsPass,
                pass_spec=None,
            ),
            AvailablePass(
                display_name="test-lower-memref-stream-to-snitch-stream",
                module_pass=test_lower_memref_stream_to_snitch_stream.TestLowerMemrefStreamToSnitchStream,
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
