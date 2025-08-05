from collections.abc import Sequence

from xdsl.dialects.builtin import ModuleOp
from xdsl.interactive.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.traits import HasCanonicalizationPatternsTrait


def get_all_possible_rewrites(module: ModuleOp) -> Sequence[ModulePass]:
    """
    Function that takes a sequence of IndividualRewrite Patterns and a ModuleOp, and
    returns the possible rewrites.
    Issue filed: https://github.com/xdslproject/xdsl/issues/2162
    """
    from xdsl.transforms import individual_rewrite

    res: list[individual_rewrite.ApplyIndividualRewritePass] = []

    for op_idx, matched_op in enumerate(module.walk()):
        if (trait := matched_op.get_trait(HasCanonicalizationPatternsTrait)) is None:
            continue

        pattern_by_name = {
            type(pattern).__name__: pattern
            for pattern in trait.get_canonicalization_patterns()
        }

        for pattern_name, pattern in pattern_by_name.items():
            cloned_op = tuple(module.clone().walk())[op_idx]
            rewriter = PatternRewriter(cloned_op)
            pattern.match_and_rewrite(cloned_op, rewriter)
            if rewriter.has_done_action:
                res.append(
                    individual_rewrite.ApplyIndividualRewritePass(
                        op_idx, cloned_op.name, pattern_name
                    ),
                )

    return res
