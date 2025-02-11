from collections.abc import Callable

from xdsl.interactive.passes import (
    AvailablePass,
    apply_passes_to_module,
    get_condensed_pass_list,
    get_new_registered_context,
)
from xdsl.interactive.rewrites import get_all_possible_rewrites
from xdsl.ir import Dialect
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import RewritePattern


def get_available_pass_list(
    all_dialects: tuple[tuple[str, Callable[[], Dialect]], ...],
    all_passes: tuple[tuple[str, type[ModulePass]], ...],
    input_text: str,
    pass_pipeline: tuple[ModulePass, ...],
    condense_mode: bool,
    rewrite_by_names_dict: dict[str, dict[str, RewritePattern]],
) -> tuple[AvailablePass, ...]:
    """
    This function returns the available pass list file based on an input text string, pass_pipeline and condense_mode.
    """
    ctx = get_new_registered_context(all_dialects)
    parser = Parser(ctx, input_text)
    current_module = parser.parse_module()

    current_module = apply_passes_to_module(current_module, ctx, pass_pipeline)

    # get all individual rewrites
    individual_rewrites = get_all_possible_rewrites(
        current_module,
        rewrite_by_names_dict,
    )
    # merge rewrite passes with "other" pass list
    if condense_mode:
        pass_list = get_condensed_pass_list(current_module, all_passes)
    else:
        pass_list = tuple(AvailablePass(p.name, p) for _, p in all_passes)
    return pass_list + tuple(individual_rewrites)
