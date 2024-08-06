from typing import cast

from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class AddImmediates(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter) -> None:
        lhs = _get_constant_int(op.lhs)
        rhs = _get_constant_int(op.rhs)
        if lhs is None:
            if rhs is None:
                pass
            else:
                if not rhs.value.data:
                    rewriter.replace_matched_op((), (op.lhs,))
        else:
            if rhs is None:
                if lhs.value.data:
                    rewriter.replace_matched_op(
                        arith.Addi(op.rhs, op.lhs, result_type=op.result.type)
                    )
                else:
                    rewriter.replace_matched_op((), (op.rhs,))
            else:
                rewriter.replace_matched_op(
                    arith.Constant(
                        IntegerAttr(lhs.value.data + rhs.value.data, lhs.type)
                    )
                )


class MultiplyImmediates(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Muli, rewriter: PatternRewriter) -> None:
        lhs = _get_constant_int(op.lhs)
        rhs = _get_constant_int(op.rhs)
        if lhs is None:
            if rhs is None:
                pass
            else:
                if rhs.value.data == 0:
                    rewriter.replace_matched_op(
                        arith.Constant(IntegerAttr(0, rhs.type))
                    )
                elif rhs.value.data == 1:
                    rewriter.replace_matched_op((), (op.lhs,))
                else:
                    pass
        else:
            if rhs is None:
                if lhs.value.data == 0:
                    rewriter.replace_matched_op(
                        arith.Constant(IntegerAttr(0, lhs.type))
                    )
                elif lhs.value.data == 1:
                    rewriter.replace_matched_op((), (op.rhs,))
                else:
                    rewriter.replace_matched_op(
                        arith.Muli(op.rhs, op.lhs, result_type=op.result.type)
                    )
            else:
                rewriter.replace_matched_op(
                    arith.Constant(
                        IntegerAttr(lhs.value.data * rhs.value.data, lhs.type)
                    )
                )


def _get_constant_int(ssa_value: SSAValue):
    if not isinstance(ssa_value.owner, arith.Constant):
        return None
    integet_attr = cast(IntegerAttr[IntegerType | IndexType], ssa_value.owner.value)
    return integet_attr
