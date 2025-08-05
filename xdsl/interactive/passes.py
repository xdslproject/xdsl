from collections.abc import Callable
from typing import NamedTuple

from xdsl.context import Context
from xdsl.dialects import builtin, get_all_dialects
from xdsl.ir import Dialect
from xdsl.passes import ModulePass
from xdsl.transforms.mlir_opt import MLIROptPass


class AvailablePass(NamedTuple):
    """
    Type alias for the attributes that describe a pass, namely the display name of the
    pass, the module pass and pass spec.
    """

    module_pass: type[ModulePass] | ModulePass

    def __str__(self) -> str:
        module_pass = self.module_pass
        if isinstance(module_pass, ModulePass):
            return str(module_pass.pipeline_pass_spec())
        else:
            return module_pass.name


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
            yield AvailablePass(pass_type)
            continue
        for p in pass_type.applicable_params(ctx, input):
            yield AvailablePass(p)


def get_condensed_pass_list(
    input: builtin.ModuleOp,
    all_passes: tuple[tuple[str, type[ModulePass]], ...],
) -> tuple[AvailablePass, ...]:
    """
    Function that returns the condensed pass list for a given ModuleOp, i.e. the passes that
    change the ModuleOp.
    """
    return tuple(iter_condensed_passes(input, all_passes))
