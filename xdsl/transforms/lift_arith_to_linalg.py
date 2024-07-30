from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import arith, linalg, tensor
from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa


def get_or_create_tensor_shape(
    tensor_shape: TensorType[Attribute], op: Operation, rewriter: PatternRewriter
) -> SSAValue:
    """
    Looks for a tensor of the specified shape in one of two places or inserts a new one at the beginning of the region.
    """

    # check if op has a named linalg operand with a matching tensor in `outs`
    for operand in op.operands:
        if isinstance(operand, linalg.NamedOpBase):
            for out in operand.outputs:
                if out == tensor_shape:
                    return out

    # check if a tensor.empty can be found in the first block of the region
    # loop until we encounter an op that is not tensor.empty,
    # as this pass always places tensor.empty first in the region
    if (p_region := op.parent_region()) and p_region.first_block:
        curr_op = p_region.first_block.first_op
        while curr_op and isinstance(curr_op, tensor.EmptyOp):
            if curr_op.tensor.type == tensor_shape:
                return curr_op.tensor
            curr_op = curr_op.next_op

        # insert a new tensor.empty if none can be found
        rewriter.insert_op(
            new_op := tensor.EmptyOp((), tensor_shape),
            InsertPoint.at_start(p_region.first_block),
        )
        return new_op.tensor

    assert False, "Op must be inside a region with a block"


class LiftAddfPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addf, rewriter: PatternRewriter, /):
        if isa(op.result.type, TensorType[Attribute]):
            out = get_or_create_tensor_shape(op.result.type, op, rewriter)
            rewriter.replace_matched_op(
                linalg.AddOp(op.operands, [out], [op.result.type])
            )


class LiftSubfPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Subf, rewriter: PatternRewriter, /):
        if isa(op.result.type, TensorType[Attribute]):
            out = get_or_create_tensor_shape(op.result.type, op, rewriter)
            rewriter.replace_matched_op(
                linalg.SubOp(op.operands, [out], [op.result.type])
            )


class LiftMulfPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Mulf, rewriter: PatternRewriter, /):
        if isa(op.result.type, TensorType[Attribute]):
            out = get_or_create_tensor_shape(op.result.type, op, rewriter)
            rewriter.replace_matched_op(
                linalg.MulOp(op.operands, [out], [op.result.type])
            )


@dataclass(frozen=True)
class LiftArithToLinalg(ModulePass):
    name = "lift-arith-to-linalg"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LiftAddfPass(),
                    LiftSubfPass(),
                    LiftMulfPass(),
                ]
            ),
            walk_reverse=False,
            apply_recursively=False,
        )
        module_pass.rewrite_module(op)
