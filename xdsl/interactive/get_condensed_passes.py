from typing import NamedTuple

from xdsl.dialects import builtin
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.tools.command_line_tool import get_all_dialects, get_all_passes
from xdsl.transforms.mlir_opt import MLIROptPass
from xdsl.utils.parse_pipeline import PipelinePassSpec

ALL_PASSES = tuple(sorted((p_name, p()) for (p_name, p) in get_all_passes().items()))
"""Contains the list of xDSL passes."""


class AvailablePass(NamedTuple):
    """
    Type alias for the attributes that describe a pass, namely the display name of the
    pass, the module pass and pass spec.
    """

    display_name: str
    module_pass: type[ModulePass]
    pass_spec: PipelinePassSpec | None


def get_condensed_pass_list(input: builtin.ModuleOp) -> tuple[AvailablePass, ...]:
    ctx = MLContext(True)

    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)

    selections: list[AvailablePass] = []
    for _, value in ALL_PASSES:
        if value is MLIROptPass:
            # Always keep MLIROptPass as an option in condensed list
            selections.append(AvailablePass(value.name, value, None))
            continue
        try:
            cloned_module = input.clone()
            cloned_ctx = ctx.clone()
            value().apply(cloned_ctx, cloned_module)
            if input.is_structurally_equivalent(cloned_module):
                continue
        except Exception:
            pass
        selections.append(AvailablePass(value.name, value, None))

    return tuple(selections)
