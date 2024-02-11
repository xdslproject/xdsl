from typing import NamedTuple

from xdsl.dialects.builtin import ModuleOp
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.tools.command_line_tool import get_all_passes
from xdsl.transforms import individual_rewrite


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


ALL_PASSES = tuple(sorted((p_name, p()) for (p_name, p) in get_all_passes().items()))
"""Contains the list of xDSL passes."""

ALL_PATTERNS: tuple[IndividualRewrite, ...] = tuple(
    IndividualRewrite(op_name, pattern_name)
    for (op_name, pattern_by_name) in individual_rewrite.REWRITE_BY_NAMES.items()
    for (pattern_name, _) in pattern_by_name.items()
)
"""Contains all the rewrite patterns."""


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
