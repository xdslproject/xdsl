from typing import cast

from xdsl.ir import MLContext, OpResult
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    ArrayAttr,
    Float64Type,
    FloatAttr,
    ModuleOp,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    op_type_rewrite_pattern,
    RewritePattern,
    PatternRewriter,
)
from xdsl.transforms.dead_code_elimination import dce
from xdsl.utils.hints import isa

from ..dialects.toy import ConstantOp, ReshapeOp, TensorTypeF64, TransposeOp


class SimplifyRedundantTranspose(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransposeOp, rewriter: PatternRewriter):
        """
        Fold transpose(transpose(x)) -> x
        """
        # Look at the input of the current transpose.
        transpose_input = op.arg
        if not isinstance(transpose_input, OpResult):
            # Input was not produced by an operation, could be a function argument
            return

        transpose_input_op = transpose_input.op
        if not isinstance(transpose_input_op, TransposeOp):
            # Input defined by another transpose? If not, no match.
            return

        rewriter.replace_op(op, [], [transpose_input_op.arg])


class ReshapeReshapeOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReshapeOp, rewriter: PatternRewriter):
        """
        Reshape(Reshape(x)) = Reshape(x)
        """
        # Look at the input of the current reshape.
        reshape_input = op.arg
        if not isinstance(reshape_input, OpResult):
            # Input was not produced by an operation, could be a function argument
            return

        reshape_input_op = reshape_input.op
        if not isinstance(reshape_input_op, ReshapeOp):
            # Input defined by another transpose? If not, no match.
            return

        t = cast(TensorTypeF64, op.res.typ)
        new_op = ReshapeOp.from_input_and_type(reshape_input_op.arg, t)
        rewriter.replace_matched_op(new_op)


class FoldConstantReshapeOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReshapeOp, rewriter: PatternRewriter):
        """
        Reshaping a constant can be done at compile time
        """
        # Look at the input of the current reshape.
        reshape_input = op.arg
        if not isinstance(reshape_input, OpResult):
            # Input was not produced by an operation, could be a function argument
            return

        reshape_input_op = reshape_input.op
        if not isinstance(reshape_input_op, ConstantOp):
            # Input defined by another transpose? If not, no match.
            return

        assert isa(op.res.typ, TensorTypeF64)
        assert isa(reshape_input_op.value.data, ArrayAttr[FloatAttr[Float64Type]])

        new_value = DenseIntOrFPElementsAttr.create_dense_float(
            type=op.res.typ, data=reshape_input_op.value.data.data
        )
        new_op = ConstantOp(new_value)
        rewriter.replace_matched_op(new_op)


class OptimiseToy(ModulePass):
    name = "dce"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(SimplifyRedundantTranspose()).rewrite_module(op)
        PatternRewriteWalker(ReshapeReshapeOpPattern()).rewrite_module(op)
        PatternRewriteWalker(FoldConstantReshapeOpPattern()).rewrite_module(op)
        dce(op)
