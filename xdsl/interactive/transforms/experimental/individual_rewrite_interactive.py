from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp
from xdsl.interactive.transforms.canonicalization_patterns.arith import (
    arith_op_to_rewrite_pattern,
)
from xdsl.ir import MLContext
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.tools.command_line_tool import get_all_dialects
from xdsl.transforms import individual_rewrite

INTERACTIVE_REWRITE_BY_NAMES: dict[str, dict[str, RewritePattern]] = {
    op.name: {
        pattern.__class__.__name__: pattern
        for pattern in arith_op_to_rewrite_pattern[op]
    }
    for dialect in get_all_dialects().values()
    for op in dialect().operations
    if op in arith_op_to_rewrite_pattern
}
"""
Returns a dictionary representing all possible experimental interactive rewrites. Keys are operation names, and
values are dictionaries. In the inner dictionary, the keys are names of patterns
associated with each operation, and the values are the corresponding RewritePattern
instances.
"""

ALL_REWRITES: dict[str, dict[str, RewritePattern]] = {
    op_name: {
        **individual_rewrite.REWRITE_BY_NAMES.get(op_name, {}),
        **INTERACTIVE_REWRITE_BY_NAMES.get(op_name, {}),
    }
    for op_name in set(individual_rewrite.REWRITE_BY_NAMES)
    | set(INTERACTIVE_REWRITE_BY_NAMES)
}
"""
Concatenates REWRITE_BY_NAMES and INTERACTIVE_REWRITE_BY_NAMES.
"""


@dataclass
class IndividualRewriteInteractive(individual_rewrite.IndividualRewrite):
    """
    Module pass representing the application of an interactive individual rewrite pattern to a module.

    Matches the operation at the provided index within the module and applies the rewrite
    pattern specified by the operation and pattern names.
    """

    name = "apply-interactive-individual-rewrite"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        assert self.matched_operation_index is not None
        assert self.operation_name is not None
        assert self.pattern_name is not None

        matched_operation_list = list(op.walk())
        if self.matched_operation_index >= len(matched_operation_list):
            raise ValueError("Matched operation index out of range.")

        matched_operation = list(op.walk())[self.matched_operation_index]
        rewriter = PatternRewriter(matched_operation)

        rewrite_dictionary = ALL_REWRITES.get(self.operation_name)
        if rewrite_dictionary is None:
            raise ValueError(
                f"Operation name {self.operation_name} not found in the rewrite dictionary."
            )

        pattern = rewrite_dictionary.get(self.pattern_name)
        if pattern is None:
            raise ValueError(
                f"Pattern name {self.pattern_name} not found for the provided operation name."
            )

        pattern.match_and_rewrite(matched_operation, rewriter)
        if not rewriter.has_done_action:
            raise ValueError("Invalid rewrite at current location.")
