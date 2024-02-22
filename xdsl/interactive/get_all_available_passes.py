from xdsl.interactive.passes import (
    ALL_PASSES,
    AvailablePass,
    apply_passes_to_module,
    get_condensed_pass_list,
    get_new_registered_context,
)
from xdsl.interactive.rewrites import (
    convert_indexed_individual_rewrites_to_available_pass,
    get_all_possible_rewrites,
)
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import RewritePattern
from xdsl.utils.parse_pipeline import PipelinePassSpec


def get_available_pass_list(
    input_text: str,
    pass_pipeline: tuple[tuple[type[ModulePass], PipelinePassSpec], ...],
    condense_mode: bool,
    rewrite_by_names_dict: dict[str, dict[str, RewritePattern]],
) -> tuple[AvailablePass, ...]:
    """
    This function returns the available pass list file based on an input text string, pass_pipeline and condense_mode.
    """
    ctx = get_new_registered_context()
    parser = Parser(ctx, input_text)
    current_module = parser.parse_module()

    current_module = apply_passes_to_module(current_module, ctx, pass_pipeline)

    # get all rewrites
    rewrites = get_all_possible_rewrites(
        current_module,
        rewrite_by_names_dict,
    )
    # transform rewrites into passes
    rewrites_as_pass_list = convert_indexed_individual_rewrites_to_available_pass(
        rewrites, current_module
    )
    # merge rewrite passes with "other" pass list
    if condense_mode:
        pass_list = get_condensed_pass_list(current_module)
        return pass_list + rewrites_as_pass_list
    else:
        pass_list = tuple(AvailablePass(p.name, p, None) for _, p in ALL_PASSES)
        return pass_list + rewrites_as_pass_list
