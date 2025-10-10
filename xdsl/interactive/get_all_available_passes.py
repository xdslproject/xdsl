from collections.abc import Callable

from xdsl.interactive.passes import (
    AvailablePass,
    get_condensed_pass_list,
    get_new_registered_context,
)
from xdsl.ir import Dialect
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PassPipeline


def get_available_pass_list(
    all_dialects: tuple[tuple[str, Callable[[], Dialect]], ...],
    all_passes: tuple[tuple[str, type[ModulePass]], ...],
    input_text: str,
    pass_pipeline: tuple[ModulePass, ...],
    condense_mode: bool,
) -> tuple[AvailablePass, ...]:
    """
    This function returns the available pass list file based on an input text string, pass_pipeline and condense_mode.
    """
    ctx = get_new_registered_context(all_dialects)
    parser = Parser(ctx, input_text)
    current_module = parser.parse_module()

    PassPipeline(pass_pipeline).apply(ctx, current_module)

    # merge rewrite passes with "other" pass list
    if condense_mode:
        pass_list = get_condensed_pass_list(ctx, current_module, all_passes)
    else:
        pass_list = tuple(AvailablePass(p) for _, p in all_passes)
    return pass_list
