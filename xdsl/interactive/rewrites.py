from collections.abc import Sequence
from typing import NamedTuple

from xdsl.dialects.builtin import ModuleOp
from xdsl.interactive.passes import AvailablePass
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.transforms import individual_rewrite
from xdsl.utils.parse_pipeline import PipelinePassSpec


class IndividualRewrite(NamedTuple):
    """
    Type alias for a possible rewrite, described by an operation and pattern name.
    """

    operation: str
    pattern: str


class IndexedIndividualRewrite(NamedTuple):
    """
    Type alias for a specific rewrite pattern, additionally consisting of its operation index.
    """

    operation_index: int
    rewrite: IndividualRewrite


def convert_indexed_individual_rewrites_to_available_pass(
    rewrites: Sequence[IndexedIndividualRewrite], current_module: ModuleOp
) -> tuple[AvailablePass, ...]:
    """
    Function that takes a tuple of rewrites, converts each rewrite into an IndividualRewrite pass and returns the tuple of AvailablePass.
    """
    rewrites_as_pass_list: tuple[AvailablePass, ...] = ()
    for op_idx, (op_name, pat_name) in rewrites:
        rewrite_pass = individual_rewrite.ApplyIndividualRewritePass
        rewrite_spec = PipelinePassSpec(
            name=rewrite_pass.name,
            args={
                "matched_operation_index": (op_idx,),
                "operation_name": (op_name,),
                "pattern_name": (pat_name,),
            },
        )
        op = list(current_module.walk())[op_idx]
        rewrites_as_pass_list = (
            *rewrites_as_pass_list,
            (AvailablePass(f"{op}:{op_name}:{pat_name}", rewrite_pass, rewrite_spec)),
        )
    return rewrites_as_pass_list


def get_all_possible_rewrites(
    module: ModuleOp,
    rewrite_by_name: dict[str, dict[str, RewritePattern]],
) -> Sequence[IndexedIndividualRewrite]:
    """
    Function that takes a sequence of IndividualRewrite Patterns and a ModuleOp, and
    returns the possible rewrites.
    Issue filed: https://github.com/xdslproject/xdsl/issues/2162
    """

    res: list[IndexedIndividualRewrite] = []

    for op_idx, matched_op in enumerate(module.walk()):
        if matched_op.name not in rewrite_by_name:
            continue
        pattern_by_name = rewrite_by_name[matched_op.name]
        for pattern_name, pattern in pattern_by_name.items():
            cloned_op = tuple(module.clone().walk())[op_idx]
            rewriter = PatternRewriter(cloned_op)
            pattern.match_and_rewrite(cloned_op, rewriter)
            if rewriter.has_done_action:
                res.append(
                    IndexedIndividualRewrite(
                        op_idx, IndividualRewrite(cloned_op.name, pattern_name)
                    )
                )

    return res
