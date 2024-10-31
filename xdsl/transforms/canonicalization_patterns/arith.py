from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import IntegerAttr
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


class AddImmediateZero(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.lhs.owner, arith.Constant)
            and isinstance(value := op.lhs.owner.value, IntegerAttr)
            and value.value.data == 0
        ):
            rewriter.replace_matched_op([], [op.rhs])


class FoldConstConstOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.FloatingPointLikeBinaryOperation, rewriter: PatternRewriter, /
    ):
        if (
            isinstance(op.lhs.owner, arith.Constant)
            and isinstance(op.rhs.owner, arith.Constant)
            and isa(l := op.lhs.owner.value, builtin.AnyFloatAttr)
            and isa(r := op.rhs.owner.value, builtin.AnyFloatAttr)
        ):
            match type(op):
                case arith.Addf:
                    val = l.value.data + r.value.data
                case arith.Subf:
                    val = l.value.data - r.value.data
                case arith.Mulf:
                    val = l.value.data * r.value.data
                case arith.Divf:
                    if r.value.data == 0.0:
                        # this mirrors what mlir does
                        val = float("inf")
                    else:
                        val = l.value.data / r.value.data
                case _:
                    return
            rewriter.replace_matched_op(arith.Constant(builtin.FloatAttr(val, l.type)))
