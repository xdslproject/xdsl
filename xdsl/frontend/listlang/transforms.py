from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.frontend.listlang import list_dialect
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


class LengthOfMap(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: list_dialect.LengthOp, rewriter: PatternRewriter):
        if not isa(op.li.owner, list_dialect.MapOp):
            return

        rewriter.replace_op(op, list_dialect.LengthOp(op.li.owner.li))


class MapOfMap(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: list_dialect.MapOp, rewriter: PatternRewriter):
        if not isa(op.li.owner, list_dialect.MapOp):
            return

        if op.li.uses.get_length() != 1:
            return

        first_block = op.li.owner.body.block
        second_block = op.body.block

        assert isa(first_block.last_op, list_dialect.YieldOp)
        first_block_result = first_block.last_op.yielded
        rewriter.erase_op(first_block.last_op)

        rewriter.inline_block(
            second_block, InsertPoint.at_end(first_block), [first_block_result]
        )

        rewriter.replace_value_with_new_type(op.li.owner.result, op.result.type)

        rewriter.replace_all_uses_with(op.result, op.li)
        rewriter.erase_op(op)


class OptimizeListOps(ModulePass):
    """
    Applies optimizations to list operations.
    """

    name = "optimize-lists"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LengthOfMap(),
                    MapOfMap(),
                ]
            ),
        ).rewrite_module(op)
