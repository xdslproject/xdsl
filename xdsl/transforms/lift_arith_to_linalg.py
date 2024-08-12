from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import arith, bufferization, linalg, tensor
from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.ir import Attribute, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


def get_dps_arg(
    op: arith.BinaryOperation[Attribute], typ: TensorType[Attribute]
) -> SSAValue:
    """
    Helper function to find a suitable `outs` tensor for DPS-style calls esp. in partially bufferized programs.

    By default, returns a tensor that was used by a linalg operand in `outs` before.

    If none is found, try backwards-traversing ops in the block to look for a `bufferization.to_tensor` with
    the `writable` flag set. Return if the shape matches (or is made to match by an `tensor.extract_slice` op).
    """

    if isinstance(op.lhs, OpResult) and isinstance(op.lhs.op, linalg.NamedOpBase):
        return op.lhs
    if isinstance(op.rhs, OpResult) and isinstance(op.rhs.op, linalg.NamedOpBase):
        return op.rhs
    curr = op
    while curr := curr.prev_op:
        if (
            isinstance(curr, bufferization.ToTensorOp)
            and curr.writable
            and curr.tensor.type == typ
        ):
            return curr.tensor
        if (
            isinstance(curr, tensor.ExtractSliceOp)
            and curr.result.type == typ
            and isinstance(curr.source, OpResult)
            and isinstance(curr.source.op, bufferization.ToTensorOp)
            and curr.source.op.writable
        ):
            return curr.result

    # fallback
    return op.lhs


class LiftAddfPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addf, rewriter: PatternRewriter, /):
        if isa(op.result.type, TensorType[Attribute]):
            rewriter.replace_matched_op(
                linalg.AddOp(
                    op.operands, [get_dps_arg(op, op.result.type)], [op.result.type]
                )
            )


class LiftSubfPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Subf, rewriter: PatternRewriter, /):
        if isa(op.result.type, TensorType[Attribute]):
            rewriter.replace_matched_op(
                linalg.SubOp(
                    op.operands, [get_dps_arg(op, op.result.type)], [op.result.type]
                )
            )


class LiftMulfPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Mulf, rewriter: PatternRewriter, /):
        if isa(op.result.type, TensorType[Attribute]):
            rewriter.replace_matched_op(
                linalg.MulOp(
                    op.operands, [get_dps_arg(op, op.result.type)], [op.result.type]
                )
            )


@dataclass(frozen=True)
class LiftArithToLinalg(ModulePass):
    """
    Pass that lifts arith ops to linalg in order to make use of destination-passing style and bufferization.

    Supports partially bufferized programs and attempts to find a suitable `outs` DPS-style argument,
    either from a `writable` `bufferization.to_tensor` with a matching size or after a `tensor.extract_slice`
    causing the size to match.
    """

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
