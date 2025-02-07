from collections.abc import Sequence

from xdsl.dialects.builtin import ModuleOp
from xdsl.interactive.passes import AvailablePass
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.traits import HasCanonicalizationPatternsTrait
from xdsl.transforms import individual_rewrite


def get_all_possible_rewrites(
    module: ModuleOp,
    rewrite_by_name: dict[str, dict[str, RewritePattern]],
) -> Sequence[AvailablePass]:
    """
    Function that takes a sequence of IndividualRewrite Patterns and a ModuleOp, and
    returns the possible rewrites.
    Issue filed: https://github.com/xdslproject/xdsl/issues/2162
    """

    res: list[AvailablePass] = []

    for op_idx, matched_op in enumerate(module.walk()):
        pattern_by_name = rewrite_by_name.get(matched_op.name, {}).copy()

        if (
            trait := matched_op.get_trait(HasCanonicalizationPatternsTrait)
        ) is not None:
            for pattern in trait.get_canonicalization_patterns():
                pattern_by_name[type(pattern).__name__] = pattern

        for pattern_name, pattern in pattern_by_name.items():
            cloned_op = tuple(module.clone().walk())[op_idx]
            rewriter = PatternRewriter(cloned_op)
            pattern.match_and_rewrite(cloned_op, rewriter)
            if rewriter.has_done_action:
                res.append(
                    AvailablePass(
                        f"{cloned_op}:{cloned_op.name}:{pattern_name}",
                        individual_rewrite.ApplyIndividualRewritePass(
                            op_idx, cloned_op.name, pattern_name
                        ),
                    )
                )

    return res
