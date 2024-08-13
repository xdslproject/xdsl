from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import MLContext
from xdsl.dialects import arith, bufferization, linalg, tensor
from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.ir import Attribute, Operation, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


def get_dps_arg_simple(op: Operation, default: SSAValue) -> SSAValue:
    """Simple strategy to find `outs` tensor that trivially returns the default value."""
    return default


def get_dps_arg_partially_bufferized(op: Operation, default: SSAValue) -> SSAValue:
    """
    Find a suitable `outs` tensor for DPS-style calls in partially bufferized programs.

    By default, returns a tensor that was used by a linalg operand in `outs` before.

    If none is found, try backwards-traversing ops in the block to look for a `bufferization.to_tensor` with
    the `writable` flag set. Return if the shape matches (or is made to match by an `tensor.extract_slice` op).
    """

    assert isa(default.type, TensorType[Attribute])
    assert isa(op, arith.BinaryOperation)
    if isinstance(op.lhs, OpResult) and isinstance(op.lhs.op, linalg.NamedOpBase):
        return op.lhs
    if isinstance(op.rhs, OpResult) and isinstance(op.rhs.op, linalg.NamedOpBase):
        return op.rhs
    curr = op
    while curr := curr.prev_op:
        if (
            isinstance(curr, bufferization.ToTensorOp)
            and curr.writable
            and curr.tensor.type == default.type
        ):
            return curr.tensor
        if (
            isinstance(curr, tensor.ExtractSliceOp)
            and curr.result.type == default.type
            and isinstance(curr.source, OpResult)
            and isinstance(curr.source.op, bufferization.ToTensorOp)
            and curr.source.op.writable
        ):
            return curr.result

    # fallback
    return default


@dataclass
class LiftArithWithStrategy(RewritePattern):
    get_dps_arg: Callable[[Operation, SSAValue], SSAValue]


class LiftAddfPass(LiftArithWithStrategy):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addf, rewriter: PatternRewriter, /):
        if isa(op.result.type, TensorType[Attribute]):
            rewriter.replace_matched_op(
                linalg.AddOp(
                    op.operands, [self.get_dps_arg(op, op.lhs)], [op.result.type]
                )
            )


class LiftSubfPass(LiftArithWithStrategy):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Subf, rewriter: PatternRewriter, /):
        if isa(op.result.type, TensorType[Attribute]):
            rewriter.replace_matched_op(
                linalg.SubOp(
                    op.operands, [self.get_dps_arg(op, op.lhs)], [op.result.type]
                )
            )


class LiftMulfPass(LiftArithWithStrategy):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Mulf, rewriter: PatternRewriter, /):
        if isa(op.result.type, TensorType[Attribute]):
            rewriter.replace_matched_op(
                linalg.MulOp(
                    op.operands, [self.get_dps_arg(op, op.lhs)], [op.result.type]
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

    STRATEGIES: ClassVar[dict[str, Callable[[Operation, SSAValue], SSAValue]]] = {
        "simple": get_dps_arg_simple,
        "from_partially_bufferized": get_dps_arg_partially_bufferized,
    }

    strategy: str = "simple"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        assert self.strategy in self.STRATEGIES
        strategy = self.STRATEGIES[self.strategy]
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LiftAddfPass(strategy),
                    LiftSubfPass(strategy),
                    LiftMulfPass(strategy),
                ]
            ),
            walk_reverse=False,
            apply_recursively=False,
        )
        module_pass.rewrite_module(op)
