from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects import arith
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
    def match_and_rewrite(self, op: arith.AddiOp, rewriter: PatternRewriter) -> None:
        if op.lhs == op.rhs:
            assert isinstance(type := op.lhs.type, IntegerType | IndexType)
            rewriter.replace_matched_op(
                [
                    li_op := arith.ConstantOp(IntegerAttr(2, type, truncate_bits=True)),
                    arith.MuliOp(op.lhs, li_op),
                ]
            )


class DivisionOfSameVariableToOne(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.DivUIOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(mul_op := op.lhs.owner, Operation)
            and isinstance(mul_op, arith.MuliOp)
            and (op.lhs in mul_op.results)
            # and mul_op.rhs == op.rhs
            and isinstance(mul_op.rhs.owner, arith.ConstantOp)
            and isinstance(mul_rhs_value := mul_op.rhs.owner.value, IntegerAttr)
            and isinstance(op.rhs.owner, arith.ConstantOp)
            and isinstance(value := op.rhs.owner.value, IntegerAttr)
            and mul_rhs_value.value.data == value.value.data
            and value.value.data != 0
        ):
            rewriter.replace_matched_op([], [mul_op.lhs])


INDIVIDUAL_REWRITE_PATTERNS_BY_NAME: dict[str, dict[str, RewritePattern]] = {
    arith.AddiOp.name: {
        AdditionOfSameVariablesToMultiplyByTwo.__name__: AdditionOfSameVariablesToMultiplyByTwo()
    },
    arith.DivUIOp.name: {
        DivisionOfSameVariableToOne.__name__: DivisionOfSameVariableToOne()
    },
}
"""
Extra rewrite patterns available to ApplyIndividualRewritePass
"""


def _get_canonicalization_pattern(
    op: Operation, pattern_name: str
) -> RewritePattern | None:
    if (trait := op.get_trait(HasCanonicalizationPatternsTrait)) is None:
        return None

    for pattern in trait.get_canonicalization_patterns():
        if type(pattern).__name__ == pattern_name:
            return pattern


@dataclass(frozen=True)
class ApplyIndividualRewritePass(ModulePass):
    """
    Module pass representing the application of an individual rewrite pattern to a module.

    Matches the operation at the provided index within the module and applies the rewrite
    pattern specified by the operation and pattern names.
    """

    name = "apply-individual-rewrite"

    matched_operation_index: int = field()
    operation_name: str = field()
    pattern_name: str = field()

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        all_ops = list(op.walk())
        if self.matched_operation_index >= len(all_ops):
            raise ValueError("Matched operation index out of range.")

        matched_operation = all_ops[self.matched_operation_index]
        rewriter = PatternRewriter(matched_operation)

        if matched_operation.name != self.operation_name:
            raise ValueError(
                f"Operation {matched_operation.name} at index "
                f"{self.matched_operation_index} does not match {self.operation_name}"
            )

        # Check individual rewrites first
        if (
            individual_rewrites := INDIVIDUAL_REWRITE_PATTERNS_BY_NAME.get(
                self.operation_name
            )
        ) is not None and (p := individual_rewrites.get(self.pattern_name)) is not None:
            pattern = p
        elif (
            p := _get_canonicalization_pattern(matched_operation, self.pattern_name)
        ) is not None:
            pattern = p
        else:
            raise ValueError(
                f"Pattern name {self.pattern_name} not found for the provided operation name."
            )

        pattern.match_and_rewrite(matched_operation, rewriter)
        if not rewriter.has_done_action:
            raise ValueError(
                f"Invalid rewrite ({self.pattern_name}) for operation "
                f"({matched_operation}) at location {self.matched_operation_index}."
            )
