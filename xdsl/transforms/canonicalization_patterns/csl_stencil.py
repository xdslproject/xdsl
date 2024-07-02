from xdsl.dialects import tensor
from xdsl.dialects.csl import csl_stencil
from xdsl.ir import OpResult
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class RedundantIterArgInitialisation(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: csl_stencil.ApplyOp, rewriter: PatternRewriter
    ) -> None:
        if len(op.iter_arg.uses) > 1:
            return

        next_apply = op
        while (next_apply := next_apply.next_op) is not None:
            if (
                isinstance(next_apply, csl_stencil.ApplyOp)
                and len(next_apply.iter_arg.uses) == 1
                and isinstance(next_apply.iter_arg, OpResult)
                and isinstance(next_apply.iter_arg.op, tensor.EmptyOp)
            ):
                rewriter.replace_op(next_apply.iter_arg.op, [], [op.iter_arg])
