from typing import NamedTuple

from xdsl.context import MLContext
from xdsl.dialects import builtin, get_all_dialects
from xdsl.passes import ModulePass, PipelinePass
from xdsl.tools.command_line_tool import get_all_passes
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


def get_new_registered_context() -> MLContext:
    """
    Generates a new MLContext, registers it and returns it.
    """
    ctx = MLContext(True)
    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)
    return ctx


def apply_passes_to_module(
    module: builtin.ModuleOp,
    ctx: MLContext,
    pass_pipeline: tuple[tuple[type[ModulePass], PipelinePassSpec], ...],
) -> builtin.ModuleOp:
    """
    Function that takes a ModuleOp, an MLContext and a pass_pipeline (consisting of a type[ModulePass] and PipelinePassSpec), applies the pass(es) to the ModuleOp and returns the new ModuleOp.
    """
    pipeline = PipelinePass(
        passes=tuple(
            module_pass.from_pass_spec(pipeline_pass_spec)
            for module_pass, pipeline_pass_spec in pass_pipeline
        )
    )
    pipeline.apply(ctx, module)
    return module


def iter_condensed_passes(input: builtin.ModuleOp):
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
            yield AvailablePass(value.name, value, None), cloned_module
        except Exception:
            pass


def get_condensed_pass_list(input: builtin.ModuleOp) -> tuple[AvailablePass, ...]:
    """
    Function that returns the condensed pass list for a given ModuleOp, i.e. the passes that
    change the ModuleOp.
    """
    return tuple(ap for ap, _ in iter_condensed_passes(input))
