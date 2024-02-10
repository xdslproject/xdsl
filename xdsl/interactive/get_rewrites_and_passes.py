from typing import NamedTuple

from xdsl.dialects import builtin
from xdsl.dialects.builtin import ModuleOp
from xdsl.interactive.app import ALL_PASSES
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.tools.command_line_tool import get_all_dialects
from xdsl.transforms import individual_rewrite
from xdsl.transforms.mlir_opt import MLIROptPass


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
    patterns: tuple[IndividualRewrite, ...],
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
        if matched_op.name not in individual_rewrite.REWRITE_BY_NAMES:
            continue
        pattern_by_name = individual_rewrite.REWRITE_BY_NAMES[matched_op.name]

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
                matched_op = list(current_module.walk())[op_idx]

    return res


def condensed_pass_list(input: builtin.ModuleOp) -> tuple[type[ModulePass], ...]:
    """Returns a tuple of passes (pass name and pass instance) that modify the IR."""

    ctx = MLContext(True)

    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)

    selections: list[type[ModulePass]] = []
    for _, value in ALL_PASSES:
        if value is MLIROptPass:
            # Always keep MLIROptPass as an option in condensed list
            selections.append(value)
            continue
        try:
            cloned_module = input.clone()
            cloned_ctx = ctx.clone()
            value().apply(cloned_ctx, cloned_module)
            if input.is_structurally_equivalent(cloned_module):
                continue
        except Exception:
            pass
        selections.append(value)

    return tuple(selections)
