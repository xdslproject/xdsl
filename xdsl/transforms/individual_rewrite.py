from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp
from xdsl.interactive.canonicalization_patterns.arith import (
    get_interactive_arith_canonicalization_patterns,
    has_interactive_canonicalization_pattern,
)
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.tools.command_line_tool import get_all_dialects
from xdsl.traits import HasCanonicalisationPatternsTrait

REWRITE_BY_NAMES: dict[str, dict[str, RewritePattern]] = {
    op.name: {
        pattern.__class__.__name__: pattern
        for pattern in trait.get_canonicalization_patterns()
    }
    for dialect in get_all_dialects().values()
    for op in dialect().operations
    if (trait := op.get_trait(HasCanonicalisationPatternsTrait)) is not None
}
"""
Returns a dictionary representing all possible rewrites. Keys are operation names, and
values are dictionaries. In the inner dictionary, the keys are names of patterns
associated with each operation, and the values are the corresponding RewritePattern
instances.
"""


INTERACTIVE_REWRITE_BY_NAMES: dict[str, dict[str, RewritePattern]] = {
    op.name: {
        pattern.__class__.__name__: pattern
        for pattern in get_interactive_arith_canonicalization_patterns()
    }
    for dialect in get_all_dialects().values()
    for op in dialect().operations
    if has_interactive_canonicalization_pattern(op)
}
"""
Returns a dictionary representing all possible experimental interactive rewrites. Keys are operation names, and
values are dictionaries. In the inner dictionary, the keys are names of patterns
associated with each operation, and the values are the corresponding RewritePattern
instances.
"""

ALL_REWRITES: dict[str, dict[str, RewritePattern]] = {
    op_name: {
        **REWRITE_BY_NAMES.get(op_name, {}),
        **INTERACTIVE_REWRITE_BY_NAMES.get(op_name, {}),
    }
    for op_name in set(REWRITE_BY_NAMES) | set(INTERACTIVE_REWRITE_BY_NAMES)
}


@dataclass
class IndividualRewrite(ModulePass):
    """
    Module pass representing the application of an individual rewrite pattern to a module.

    Matches the operation at the provided index within the module and applies the rewrite
    pattern specified by the operation and pattern names.
    """

    name = "apply-individual-rewrite"

    matched_operation_index: int | None = None
    operation_name: str | None = None
    pattern_name: str | None = None

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
