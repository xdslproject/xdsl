from xdsl.dialects import arith, complex
from xdsl.dialects.builtin import (
    AnyFloat,
    ArrayAttr,
    ComplexType,
    Float16Type,
    Float32Type,
    Float64Type,
    FloatAttr,
    IndexType,
    IntegerType,
)
from xdsl.irdl import Operation
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


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
        if (attr := op.fold()) is not None:
            rewriter.replace_matched_op(complex.ConstantOp(attr[0], op.result.type))


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


class ReRedundantOpPattern(RewritePattern):
    """
    %x = (complex.constant [a, b]) | (complex.create %c, %d)
    %y = complex.re %x = (%c | a)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: complex.ReOp, rewriter: PatternRewriter):
        if isinstance(op.complex.owner, complex.ConstantOp) and isa(
            val := op.complex.owner.value, ArrayAttr[FloatAttr]
        ):
            rewriter.replace_matched_op(arith.ConstantOp(val.data[0]))
            return
        elif (
            isinstance(operand := op.complex.owner, complex.CreateOp)
            and isinstance(l := operand.real.owner, arith.ConstantOp)
            and isinstance(l.value, FloatAttr)
        ):
            rewriter.replace_matched_op((), (operand.real,))


class ImRedundantOpPattern(RewritePattern):
    """
    %x = (complex.constant [a, b]) | (complex.create %c, %d)
    %y = complex.im %x = (%d | b)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: complex.ImOp, rewriter: PatternRewriter):
        if isinstance(op.complex.owner, complex.ConstantOp) and isa(
            val := op.complex.owner.value, ArrayAttr[FloatAttr]
        ):
            rewriter.replace_matched_op(arith.ConstantOp(val.data[1]))
            return
        elif (
            isinstance(operand := op.complex.owner, complex.CreateOp)
            and isinstance(r := operand.imaginary.owner, arith.ConstantOp)
            and isinstance(r.value, FloatAttr)
        ):
            rewriter.replace_matched_op((), (operand.imaginary,))


class ReNegOpPattern(RewritePattern):
    """
    %x = complex.create %a, %b
    %y = complex.neg %x
    %re = complex.re %y = arith.negf %a
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: complex.ReOp, rewriter: PatternRewriter):
        if isinstance(inner_op := op.complex.owner, complex.NegOp) and isinstance(
            creat_op := inner_op.complex.owner, complex.CreateOp
        ):
            rewriter.replace_matched_op(arith.NegfOp(creat_op.real))


class ImNegOpPattern(RewritePattern):
    """
    %x = complex.create %a, %b
    %y = complex.neg %x
    %im = complex.im %y = arith.negf %b
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: complex.ImOp, rewriter: PatternRewriter):
        if isinstance(inner_op := op.complex.owner, complex.NegOp) and isinstance(
            creat_op := inner_op.complex.owner, complex.CreateOp
        ):
            rewriter.replace_matched_op(arith.NegfOp(creat_op.imaginary))


class RedundantUnaryOpOpPattern(RewritePattern):
    """
    conj(conj(x)) = x
    log(exp(x)) = x
    exp(log(x)) = x
    neg(neg(x)) = x
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: complex.ComplexUnaryComplexResultOperation, rewriter: PatternRewriter
    ):
        if isinstance(op.complex.owner, Operation) and (
            (
                isa(inner_op := op.complex.owner, complex.ConjOp | complex.NegOp)
                and isinstance(inner_op, type(op))
            )
            or (isinstance(inner_op, complex.LogOp) and isinstance(op, complex.ExpOp))
            or (isinstance(inner_op, complex.ExpOp) and isinstance(op, complex.LogOp))
        ):
            rewriter.replace_matched_op((), (inner_op.complex,))


class AddSubOpPattern(RewritePattern):
    """
    %sub = complex.sub %x, %y
    %add = (complex.add %sub, %y) | (complex.add %y, %sub) = %x
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: complex.AddOp, rewriter: PatternRewriter):
        if isinstance(lhs_op := op.lhs.owner, complex.SubOp) and (lhs_op.rhs == op.rhs):
            rewriter.replace_matched_op((), (lhs_op.lhs,))
            return
        elif isinstance(rhs_op := op.rhs.owner, complex.SubOp) and (
            rhs_op.rhs == op.lhs
        ):
            rewriter.replace_matched_op((), (rhs_op.lhs,))


class SubAddOpPattern(RewritePattern):
    """
    %add = complex.add %x, %y
    %sub = complex.sub %add, %y = %x
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: complex.SubOp, rewriter: PatternRewriter):
        if isinstance(inner_op := op.lhs.owner, complex.AddOp) and (
            inner_op.rhs == op.rhs
        ):
            rewriter.replace_matched_op((), (inner_op.lhs,))


class BitcastOpPattern(RewritePattern):
    """
    %0 = complex.bitcast %arg0 : i64 to complex<f32>
    %1 = complex.bitcast %0 : complex<f32> to i64 = %arg0
    -----------------------------------------------------------------------------------------
    %0 = complex.bitcast %arg0 : f64 to complex<f32>
    %1 = complex.bitcast %0 : complex<f32> to i64 = arith.bitcast %arg0
    ------------------------------------------------------------------------------------------
    %0 = arith.bitcast %arg0 : f64 to i64
    %1 = complex.bitcast %0 : i64 to complex<f32> = complex.bitcast %arg0 : f64 to complex<f32>
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: complex.BitcastOp, rewriter: PatternRewriter):
        if isinstance(inner_op := op.operand.owner, complex.BitcastOp):
            if isinstance(
                res_type := op.result.type,
                IntegerType
                | IndexType
                | Float16Type
                | Float32Type
                | Float64Type
                | ComplexType,
            ) and isinstance(
                operand_type := inner_op.operand.type,
                IntegerType
                | IndexType
                | Float16Type
                | Float32Type
                | Float64Type
                | ComplexType,
            ):
                if res_type == operand_type:
                    rewriter.replace_matched_op((), (inner_op.operand,))
                else:
                    rewriter.replace_matched_op(
                        arith.BitcastOp(inner_op.operand, res_type)
                    )
                return
        if isinstance(inner_op := op.operand.owner, arith.BitcastOp) and isinstance(
            inner_op.input.type, AnyFloat
        ):
            rewriter.replace_matched_op(
                complex.BitcastOp(inner_op.input, op.result.type)
            )
