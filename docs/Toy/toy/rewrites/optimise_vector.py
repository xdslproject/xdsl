from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext, OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    op_type_rewrite_pattern,
    RewritePattern,
    PatternRewriter,
)
from xdsl.transforms.dead_code_elimination import dce

from ..dialects import vector as vd


class SimplifyRedundantShapeAccess(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vd.TensorShapeOp, rewriter: PatternRewriter):
        """
        Fold tensor(t_shape, t_data).data -> t_data
        """
        # Look at the input of the current transpose.
        tensor_data_input = op.tensor
        if not isinstance(tensor_data_input, OpResult):
            # Input was not produced by an operation, could be a function argument
            return

        tensor_make_op = tensor_data_input.op
        if not isinstance(tensor_make_op, vd.TensorMakeOp):
            # Input defined by a constant passed in? If not, no match.
            return

        rewriter.replace_matched_op([], [tensor_make_op.shape])


class SimplifyRedundantDataAccess(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vd.TensorDataOp, rewriter: PatternRewriter):
        """
        Fold tensor(t_shape, t_data).data -> t_data
        """
        # Look at the input of the current transpose.
        tensor_data_input = op.tensor
        if not isinstance(tensor_data_input, OpResult):
            # Input was not produced by an operation, could be a function argument
            return

        tensor_make_op = tensor_data_input.op
        if not isinstance(tensor_make_op, vd.TensorMakeOp):
            # Input defined by a constant passed in? If not, no match.
            return
        rewriter.replace_matched_op([], [tensor_make_op.data])


class OptimiseVector(ModulePass):
    name = "optimise-vector"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(SimplifyRedundantShapeAccess()).rewrite_module(op)
        PatternRewriteWalker(SimplifyRedundantDataAccess()).rewrite_module(op)
        dce(op)
