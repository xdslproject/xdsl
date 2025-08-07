from collections.abc import Sequence

from xdsl.dialects.builtin import ModuleOp
from xdsl.interactive.passes import AvailablePass
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.traits import HasCanonicalizationPatternsTrait
from xdsl.transforms import individual_rewrite
from xdsl.utils.op_selector import OpSelector


def get_all_possible_rewrites(module: ModuleOp) -> Sequence[AvailablePass]:
    """
    Function that takes a sequence of IndividualRewrite Patterns and a ModuleOp, and
    returns the possible rewrites.
    Issue filed: https://github.com/xdslproject/xdsl/issues/2162
    """

    res: list[AvailablePass] = []

    for op_idx, matched_op in enumerate(module.walk()):
        if (trait := matched_op.get_trait(HasCanonicalizationPatternsTrait)) is None:
            continue

        pattern_by_name = {
            type(pattern).__name__: pattern
            for pattern in trait.get_canonicalization_patterns()
        }

        selector = OpSelector(op_idx, matched_op.name)

        for pattern_name, pattern in pattern_by_name.items():
            cloned_op = selector.get_op(module.clone())
            rewriter = PatternRewriter(cloned_op)
            pattern.match_and_rewrite(cloned_op, rewriter)
            if rewriter.has_done_action:
                res.append(
                    AvailablePass(
                        individual_rewrite.ApplyIndividualRewritePass(
                            op_idx, cloned_op.name, pattern_name
                        ),
                    )
                )

    return res
