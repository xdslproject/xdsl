from xdsl.dialects import arith, complex
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
)
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


def _fold_const_operation(
    op_t: type[complex.ComplexBinaryOp],
    lhs: ArrayAttr[FloatAttr],
    rhs: ArrayAttr[FloatAttr],
) -> complex.ConstantOp | None:
    re_lhs, im_lhs = lhs.data[0].value.data, lhs.data[1].value.data
    re_rhs, im_rhs = rhs.data[0].value.data, rhs.data[1].value.data
    match op_t:
        case complex.AddOp:
            real = re_lhs + re_rhs
            imag = im_lhs + im_rhs
        case complex.SubOp:
            real = re_lhs - re_rhs
            imag = im_lhs - im_rhs
        case complex.MulOp:
            real = re_lhs * re_rhs - im_lhs * im_rhs
            imag = re_lhs * im_rhs + im_lhs * re_rhs
        case complex.DivOp:
            if re_rhs == 0.0 and im_rhs == 0.0:
                if re_lhs == 0.0:
                    real = float("nan")
                else:
                    real = float("inf") if re_lhs > 0 else float("-inf")
                if im_lhs == 0.0:
                    imag = float("nan")
                else:
                    imag = float("inf") if im_lhs > 0 else float("-inf")
            else:
                real = (re_lhs * re_rhs + im_lhs * im_rhs) / (re_rhs**2 + im_rhs**2)
                imag = (im_lhs * re_rhs - re_lhs * im_rhs) / (re_rhs**2 + im_rhs**2)
        case _:
            return
    return complex.ConstantOp.from_tuple_and_width(
        (real, imag), lhs.data[0].type.bitwidth
    )


class FoldConstConstOp(RewritePattern):
    """
    Folds a complex floating point binary op whose operands are both `complex.constant`s.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: complex.ComplexBinaryOp,
        rewriter: PatternRewriter,
    ):
        if (
            isinstance(op.lhs.owner, complex.ConstantOp)
            and isinstance(op.rhs.owner, complex.ConstantOp)
            and isa(l := op.lhs.owner.value, ArrayAttr[FloatAttr])
            and isa(r := op.rhs.owner.value, ArrayAttr[FloatAttr])
            and (cnst := _fold_const_operation(type(op), l, r))
        ):
            rewriter.replace_matched_op(cnst)


class RedundantCreateOpPattern(RewritePattern):
    """
    %real = complex.re %x
    %imag = complex.im %x
    %y = complex.create %real, %imag = %x
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: complex.CreateOp, rewriter: PatternRewriter):
        if (
            isinstance(op.real.owner, complex.ReOp)
            and isinstance(op.imaginary.owner, complex.ImOp)
            and ((op.real.owner.complex) is (op.imaginary.owner.complex))
        ):
            rewriter.replace_matched_op((), (op.real.owner.complex,))


class ReImRedundantOpPattern(RewritePattern):
    """
    %x = (complex.constant [a, b]) | (complex.create %c, %d)
    %y = complex.re %x = (%c | a)
    %y = complex.im %x = (%d | b)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: complex.ReOp | complex.ImOp, rewriter: PatternRewriter
    ):
        if isinstance(op.complex.owner, complex.ConstantOp) and isa(
            val := op.complex.owner.value, ArrayAttr[FloatAttr]
        ):
            index = 0 if isinstance(op, complex.ReOp) else 1
            rewriter.replace_matched_op(arith.ConstantOp(val.data[index]))
            return
        elif isinstance(operand := op.complex.owner, complex.CreateOp) and (
            (
                isinstance(l := operand.real.owner, arith.ConstantOp)
                and isinstance(l.value, FloatAttr)
            )
            or (
                isinstance(r := operand.imaginary.owner, arith.ConstantOp)
                and isinstance(r.value, FloatAttr)
            )
        ):
            new_ssa_value = (
                operand.real if isinstance(op, complex.ReOp) else operand.imaginary
            )
            rewriter.replace_matched_op((), (new_ssa_value,))


class ReImNegOpPattern(RewritePattern):
    """
    %x = complex.create %a, %b
    %y = complex.neg %x
    %re = complex.re %y = arith.negf %a
    %im = complex.im %y = arith.negf %b
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: complex.ReOp | complex.ImOp, rewriter: PatternRewriter
    ):
        if isinstance(inner_op := op.complex.owner, complex.NegOp) and isinstance(
            creat_op := inner_op.complex.owner, complex.CreateOp
        ):
            ssa_value = (
                creat_op.real if isinstance(op, complex.ReOp) else creat_op.imaginary
            )
            rewriter.replace_matched_op(arith.NegfOp(ssa_value))
