from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import x86
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class DS_VpbroadcastqOpScalarLoad(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: x86.DS_VpbroadcastqOp, rewriter: PatternRewriter
    ) -> None:
        if isinstance(mov_op := op.source.owner, x86.ops.DM_MovOp):
            rewriter.replace_op(
                op,
                x86.DM_VbroadcastsdOp(
                    mov_op.memory, mov_op.memory_offset, destination=op.destination.type
                ),
            )


@dataclass(frozen=True)
class X86InferBroadcast(ModulePass):
    """
    Rewrites a scalar load + broadcast to a broadcast load operation.
    """

    name = "x86-infer-broadcast"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            DS_VpbroadcastqOpScalarLoad(),
            apply_recursively=False,
        ).rewrite_module(op)
