from collections.abc import Callable
from typing import NamedTuple

from xdsl.context import Context
from xdsl.dialects import builtin, get_all_dialects
from xdsl.ir import Dialect
from xdsl.passes import ModulePass, PipelinePass
from xdsl.transforms.mlir_opt import MLIROptPass


class AvailablePass(NamedTuple):
    """
    Type alias for the attributes that describe a pass, namely the display name of the
    pass, the module pass and pass spec.
    """

    display_name: str
    module_pass: type[ModulePass] | ModulePass


def get_new_registered_context(
    all_dialects: tuple[tuple[str, Callable[[], Dialect]], ...],
) -> Context:
    """
    Generates a new Context, registers it and returns it.
    """
    ctx = Context(True)
    for dialect_name, dialect_factory in all_dialects:
        ctx.register_dialect(dialect_name, dialect_factory)
    return ctx


def apply_passes_to_module(
    module: builtin.ModuleOp,
    ctx: Context,
    passes: tuple[ModulePass, ...],
) -> builtin.ModuleOp:
    """
    Function that takes a ModuleOp, an Context and a pass_pipeline, applies the
    passes to the ModuleOp and returns the modified ModuleOp.
    """
    pipeline = PipelinePass(passes=passes)
    pipeline.apply(ctx, module)
    return module


def iter_condensed_passes(
    input: builtin.ModuleOp,
    all_passes: tuple[tuple[str, type[ModulePass]], ...],
):
    ctx = Context(True)

    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)

    for _, pass_type in all_passes:
        if pass_type is MLIROptPass:
            # Always keep MLIROptPass as an option in condensed list
            yield AvailablePass(pass_type.name, pass_type), None
            continue
        cloned_module = input.clone()
        cloned_ctx = ctx.clone()
        try:
            pass_instance = pass_type()
            pass_instance.apply(cloned_ctx, cloned_module)
            if input.is_structurally_equivalent(cloned_module):
                continue
        except Exception:
            continue
        yield AvailablePass(pass_type.name, pass_instance), cloned_module


def get_condensed_pass_list(
    input: builtin.ModuleOp,
    all_passes: tuple[tuple[str, type[ModulePass]], ...],
) -> tuple[AvailablePass, ...]:
    """
    Function that returns the condensed pass list for a given ModuleOp, i.e. the passes that
    change the ModuleOp.
    """
    return tuple(ap for ap, _ in iter_condensed_passes(input, all_passes))
