from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import BoolAttr, IndexType, IntegerType
from xdsl.ir import OpResult
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import Commutative
from xdsl.transforms.canonicalization_patterns.utils import (
    const_evaluate_operand,
    const_evaluate_operand_attribute,
)
from xdsl.utils.hints import isa


class SignlessIntegerBinaryOperationZeroOrUnitRight(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.SignlessIntegerBinaryOperation, rewriter: PatternRewriter, /
    ):
        if (rhs := const_evaluate_operand_attribute(op.rhs)) is None:
            return
        if op.is_right_zero(rhs):
            rewriter.replace_op(op, (), (op.rhs,))
        elif op.is_right_unit(rhs):
            rewriter.replace_op(op, (), (op.lhs,))


class SignlessIntegerBinaryOperationConstantProp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.SignlessIntegerBinaryOperation, rewriter: PatternRewriter, /
    ):
        if (lhs := const_evaluate_operand(op.lhs)) is None:
            return
        if (rhs := const_evaluate_operand(op.rhs)) is None:
            # Swap inputs if lhs is constant and rhs is not
            if op.has_trait(Commutative):
                rewriter.replace_op(op, op.__class__(op.rhs, op.lhs))
            return

        if (res := op.py_operation(lhs, rhs)) is None:
            return
        assert isinstance(op.result.type, IntegerType | IndexType)

        rewriter.replace_op(
            op,
            arith.ConstantOp.from_int_and_width(
                res, op.result.type, truncate_bits=True
            ),
        )


def _fold_const_operation(
    op_t: type[arith.FloatingPointLikeBinaryOperation],
    lhs: builtin.FloatAttr,
    rhs: builtin.FloatAttr,
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
            and isa(l := op.lhs.owner.value, builtin.FloatAttr)
            and isa(r := op.rhs.owner.value, builtin.FloatAttr)
            and (cnst := _fold_const_operation(type(op), l, r))
        ):
            rewriter.replace_op(op, cnst)


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
            or not isinstance(u := op.result.get_user_of_unique_use(), type(op))
            or not isinstance(
                const2 := u.lhs.owner if u.rhs == op.result else u.rhs.owner,
                arith.ConstantOp,
            )
            or arith.FastMathFlag.REASSOC not in op.fastmath.flags
            or arith.FastMathFlag.REASSOC not in u.fastmath.flags
            or not isa(c1 := const1.value, builtin.FloatAttr)
            or not isa(c2 := const2.value, builtin.FloatAttr)
        ):
            return

        if cnsts := _fold_const_operation(type(op), c1, c2):
            flags = arith.FastMathFlagsAttr(list(op.fastmath.flags | u.fastmath.flags))
            rebuild = type(op)(cnsts, val, flags)
            rewriter.replace_op(op, [cnsts, rebuild])
            rewriter.replace_op(u, [], [rebuild.result])


class SelectConstPattern(RewritePattern):
    """
    arith.select %true %x %y = %x
    arith.select %false %x %y = %y
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SelectOp, rewriter: PatternRewriter):
        const_value = const_evaluate_operand(op.cond)

        if const_value is None:
            return

        new_results = (op.lhs,) if const_value else (op.rhs,)
        rewriter.replace_op(op, (), new_results)


class SelectTrueFalsePattern(RewritePattern):
    """
    arith.select %x %true %false = %x
    arith.select %x %false %true = %x xor 1
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SelectOp, rewriter: PatternRewriter):
        if op.result.type != IntegerType(1):
            return

        if (lhs := const_evaluate_operand(op.lhs)) is None or (
            rhs := const_evaluate_operand(op.rhs)
        ) is None:
            return

        if lhs and not rhs:
            rewriter.replace_op(op, (), (op.cond,))

        if not lhs and rhs:
            rewriter.replace_op(op, arith.XOrIOp(op.cond, op.rhs))


class SelectSamePattern(RewritePattern):
    """
    arith.select %x %y %y = %y
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SelectOp, rewriter: PatternRewriter):
        if op.lhs == op.rhs:
            rewriter.replace_op(op, (), (op.lhs,))


class SelectFoldCmpfPattern(RewritePattern):
    """
    %1 = arith.cmpf  ogt, %0, %cst fastmath<nnan> : f64
    %2 = arith.select %1, %0, %cst : f64
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SelectOp, rewriter: PatternRewriter):
        if not isinstance(op.cond, OpResult) or not isinstance(
            cmpf := op.cond.op, arith.CmpfOp
        ):
            return
        if (
            arith.FastMathFlag.NO_NANS not in cmpf.fastmath.flags
            or arith.FastMathFlag.NO_SIGNED_ZEROS not in cmpf.fastmath.flags
        ):
            return
        if not (op.lhs == cmpf.lhs and op.rhs == cmpf.rhs):
            return

        target = None
        match cmpf.predicate.value.data:
            case 2 | 3 | 9 | 10:
                # ogt | oge | ugt | uge
                target = arith.MaximumfOp
            case 4 | 5 | 11 | 12:
                # olt | ole | ult | ule
                target = arith.MinimumfOp
            case _:
                return
        rewriter.replace_op(op, target(op.lhs, op.rhs, cmpf.fastmath))


class ApplyCmpiPredicateToEqualOperands(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.CmpiOp, rewriter: PatternRewriter):
        if op.lhs != op.rhs:
            return
        val = op.predicate.value.data in (0, 3, 5, 7, 9)
        rewriter.replace_op(op, arith.ConstantOp(BoolAttr.from_bool(val)))
