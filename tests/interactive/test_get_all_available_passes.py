from xdsl.backend.riscv.lowering import (
    convert_arith_to_riscv,
    convert_func_to_riscv_func,
)
from xdsl.dialects import get_all_dialects
from xdsl.interactive.get_all_available_passes import get_available_pass_list
from xdsl.interactive.passes import AvailablePass
from xdsl.interactive.rewrites import individual_rewrite
from xdsl.transforms import (
    get_all_passes,
    reconcile_unrealized_casts,
    test_lower_linalg_to_snitch,
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
                display_name="test-lower-linalg-to-snitch",
                module_pass=test_lower_linalg_to_snitch.TestLowerLinalgToSnitchPass,
                pass_spec=None,
            ),
        )
    )

    all_dialects = tuple((d_name, d) for d_name, d in get_all_dialects().items())
    all_passes = tuple((p_name, p()) for p_name, p in sorted(get_all_passes().items()))

    res = get_available_pass_list(
        all_dialects,
        all_passes,
        input_text,
        pass_pipeline,
        condense_mode=True,
        rewrite_by_names_dict=individual_rewrite.REWRITE_BY_NAMES,
    )

    assert res == expected_res
