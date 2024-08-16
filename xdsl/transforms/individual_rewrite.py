from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import arith, get_all_dialects
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType, ModuleOp
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import HasCanonicalizationPatternsTrait


class AdditionOfSameVariablesToMultiplyByTwo(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter) -> None:
        if op.lhs == op.rhs:
            assert isinstance(op.lhs.type, IntegerType | IndexType)
            rewriter.replace_matched_op(
                [
                    li_op := arith.Constant(IntegerAttr(2, op.lhs.type)),
                    arith.Muli(op.lhs, li_op),
                ]
            )


class DivisionOfSameVariableToOne(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.DivUI, rewriter: PatternRewriter) -> None:
        if (
            isinstance(mul_op := op.lhs.owner, Operation)
            and isinstance(mul_op, arith.Muli)
            and (op.lhs in mul_op.results)
            # and mul_op.rhs == op.rhs
            and isinstance(mul_op.rhs.owner, arith.Constant)
            and isinstance(mul_rhs_value := mul_op.rhs.owner.value, IntegerAttr)
            and isinstance(op.rhs.owner, arith.Constant)
            and isinstance(value := op.rhs.owner.value, IntegerAttr)
            and mul_rhs_value.value.data == value.value.data
            and value.value.data != 0
        ):
            rewriter.replace_matched_op([], [mul_op.lhs])


INDIVIDUAL_REWRITE_PATTERNS_BY_OP_CLASS: dict[
    type[Operation], tuple[RewritePattern, ...]
] = {
    arith.Addi: (AdditionOfSameVariablesToMultiplyByTwo(),),
    arith.DivUI: (DivisionOfSameVariableToOne(),),
}
"""
Dictionary where the key is an Operation and the value is a tuple of rewrite pattern(s) associated with that operation. These are rewrite patterns defined in this class.
"""

CANONICALIZATION_PATTERNS_BY_OP_CLASS: dict[
    type[Operation], tuple[RewritePattern, ...]
] = {
    op: trait.get_canonicalization_patterns()
    for dialect in get_all_dialects().values()
    for op in dialect().operations
    if (trait := op.get_trait(HasCanonicalizationPatternsTrait)) is not None
}
"""
Dictionary where the key is an Operation and the value is a tuple of rewrite pattern(s) associated with that operation. These are the xdsl canonicalization patterns.
"""

REWRITE_BY_NAMES: dict[str, dict[str, RewritePattern]] = {
    op.name: {
        pattern.__class__.__name__: pattern
        for pattern in INDIVIDUAL_REWRITE_PATTERNS_BY_OP_CLASS.get(op, ())
        + CANONICALIZATION_PATTERNS_BY_OP_CLASS.get(op, ())
    }
    for op in set(INDIVIDUAL_REWRITE_PATTERNS_BY_OP_CLASS)
    | set(CANONICALIZATION_PATTERNS_BY_OP_CLASS)
}
"""
Returns a dictionary representing all possible rewrites. Keys are operation names, and
values are dictionaries. In the inner dictionary, the keys are names of patterns
associated with each operation, and the values are the corresponding RewritePattern
instances.
"""


@dataclass(frozen=True)
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

        rewrite_dictionary = REWRITE_BY_NAMES.get(self.operation_name)
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
