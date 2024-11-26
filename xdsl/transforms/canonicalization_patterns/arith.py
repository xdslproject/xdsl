from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


class AddImmediateZero(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.AddiOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.lhs.owner, arith.ConstantOp)
            and isinstance(value := op.lhs.owner.value, IntegerAttr)
            and value.value.data == 0
        ):
            rewriter.replace_matched_op([], [op.rhs])


def _fold_const_operation(
    op_t: type[arith.FloatingPointLikeBinaryOperation],
    lhs: builtin.AnyFloatAttr,
    rhs: builtin.AnyFloatAttr,
) -> arith.ConstantOp | None:
    match op_t:
        case arith.AddfOp:
            val = lhs.value.data + rhs.value.data
        case arith.SubfOp:
            val = lhs.value.data - rhs.value.data
        case arith.MulfOp:
            val = lhs.value.data * rhs.value.data
        case arith.DivfOp:
            if rhs.value.data == 0.0:
                # this mirrors what mlir does
                if lhs.value.data == 0.0:
                    val = float("nan")
                elif lhs.value.data < 0:
                    val = float("-inf")
                else:
                    val = float("inf")
            else:
                val = lhs.value.data / rhs.value.data
        case _:
            return
    return arith.ConstantOp(builtin.FloatAttr(val, lhs.type))


class FoldConstConstOp(RewritePattern):
    """
    Folds a floating point binary op whose operands are both `arith.constant`s.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.FloatingPointLikeBinaryOperation, rewriter: PatternRewriter, /
    ):
        if (
            isinstance(op.lhs.owner, arith.ConstantOp)
            and isinstance(op.rhs.owner, arith.ConstantOp)
            and isa(l := op.lhs.owner.value, builtin.AnyFloatAttr)
            and isa(r := op.rhs.owner.value, builtin.AnyFloatAttr)
            and (cnst := _fold_const_operation(type(op), l, r))
        ):
            rewriter.replace_matched_op(cnst)


class FoldConstsByReassociation(RewritePattern):
    """
    Rewrites a chain of
        `(const1 <op> var) <op> const2`
    as
        `folded_const <op> val`

    The op must be associative and have the `fastmath<reassoc>` flag set.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.AddfOp | arith.MulfOp, rewriter: PatternRewriter, /
    ):
        if isinstance(op.lhs.owner, arith.ConstantOp):
            const1, val = op.lhs.owner, op.rhs
        else:
            const1, val = op.rhs.owner, op.lhs

        if (
            not isinstance(const1, arith.ConstantOp)
            or len(op.result.uses) != 1
            or not isinstance(u := list(op.result.uses)[0].operation, type(op))
            or not isinstance(
                const2 := u.lhs.owner if u.rhs == op.result else u.rhs.owner,
                arith.ConstantOp,
            )
            or op.fastmath is None
            or u.fastmath is None
            or arith.FastMathFlag.REASSOC not in op.fastmath.flags
            or arith.FastMathFlag.REASSOC not in u.fastmath.flags
            or not isa(c1 := const1.value, builtin.AnyFloatAttr)
            or not isa(c2 := const2.value, builtin.AnyFloatAttr)
        ):
            return

        if cnsts := _fold_const_operation(type(op), c1, c2):
            flags = arith.FastMathFlagsAttr(list(op.fastmath.flags | u.fastmath.flags))
            rebuild = type(op)(cnsts, val, flags)
            rewriter.replace_matched_op([cnsts, rebuild])
            rewriter.replace_op(u, [], [rebuild.result])


class SelectConstPattern(RewritePattern):
    """
    arith.select %true %x %y = %x
    arith.select %false %x %y = %y
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SelectOp, rewriter: PatternRewriter):
        if not isinstance(condition := op.cond.owner, arith.ConstantOp):
            return

        assert isinstance(const_cond := condition.value, IntegerAttr)

        if const_cond.value.data == 1:
            rewriter.replace_matched_op((), (op.lhs,))
        if const_cond.value.data == 0:
            rewriter.replace_matched_op((), (op.rhs,))


class SelectTrueFalsePattern(RewritePattern):
    """
    arith.select %x %true %false = %x
    arith.select %x %false %true = %x xor 1
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SelectOp, rewriter: PatternRewriter):
        if op.result.type != IntegerType(1):
            return

        if not isinstance(lhs := op.lhs.owner, arith.ConstantOp) or not isinstance(
            rhs := op.rhs.owner, arith.ConstantOp
        ):
            return

        assert isinstance(lhs_value := lhs.value, IntegerAttr)
        assert isinstance(rhs_value := rhs.value, IntegerAttr)

        if lhs_value.value.data == 1 and rhs_value.value.data == 0:
            rewriter.replace_matched_op((), (op.cond,))

        if lhs_value.value.data == 0 and rhs_value.value.data == 1:
            rewriter.replace_matched_op(arith.XOrIOp(op.cond, rhs))


class SelectSamePattern(RewritePattern):
    """
    arith.select %x %y %y = %y
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SelectOp, rewriter: PatternRewriter):
        if op.lhs == op.rhs:
            rewriter.replace_matched_op((), (op.lhs,))
