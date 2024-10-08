from xdsl.dialects import tensor
from xdsl.dialects.csl import csl_stencil
from xdsl.ir import OpResult
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class RedundantAccumulatorInitialisation(RewritePattern):
    """
    Removes redundant allocations of empty tensors with no uses other than passed
    as `iter_arg` to `csl_stencil.apply`. Prefer re-use where possible.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: csl_stencil.ApplyOp, rewriter: PatternRewriter
    ) -> None:
        if len(op.accumulator.uses) > 1:
            return

        next_apply = op
        while (next_apply := next_apply.next_op) is not None:
            if (
                isinstance(next_apply, csl_stencil.ApplyOp)
                and len(next_apply.accumulator.uses) == 1
                and isinstance(next_apply.accumulator, OpResult)
                and isinstance(next_apply.accumulator.op, tensor.EmptyOp)
                and op.accumulator.type == next_apply.accumulator.type
            ):
                rewriter.replace_op(next_apply.accumulator.op, [], [op.accumulator])
