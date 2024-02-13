from typing import NamedTuple

from xdsl.dialects.builtin import ModuleOp
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern


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


def get_all_possible_rewrites(
    op: ModuleOp,
    rewrite_by_name: dict[str, dict[str, RewritePattern]],
) -> tuple[IndexedIndividualRewrite, ...]:
    """
    Function that takes a sequence of IndividualRewrite Patterns and a ModuleOp, and
    returns the possible rewrites.
    """
    old_module = op.clone()
    num_ops = len(list(old_module.walk()))

    current_module = old_module.clone()

    res: tuple[IndexedIndividualRewrite, ...] = ()

    for op_idx in range(num_ops):
        matched_op = list(current_module.walk())[op_idx]
        if matched_op.name not in rewrite_by_name:
            continue
        pattern_by_name = rewrite_by_name[matched_op.name]

        for pattern_name, pattern in pattern_by_name.items():
            rewriter = PatternRewriter(matched_op)
            pattern.match_and_rewrite(matched_op, rewriter)
            if rewriter.has_done_action:
                res = (
                    *res,
                    (
                        IndexedIndividualRewrite(
                            op_idx, IndividualRewrite(matched_op.name, pattern_name)
                        )
                    ),
                )
                current_module = old_module.clone()

    return res
