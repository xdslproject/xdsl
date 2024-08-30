from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.arith import Addi, Cmpi
from xdsl.dialects.cf import Branch, ConditionalBranch
from xdsl.dialects.scf import For
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.traits import IsTerminator


class ForLowering(RewritePattern):
    """ """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, forOp: For, rewriter: PatternRewriter):
        init_block = forOp.parent_block()
        if init_block is None:
            return
        end_block = init_block.split_before(forOp)

        condition_block = forOp.body.first_block
        assert condition_block is not None
        first_op = condition_block.first_op
        assert first_op is not None
        first_body_block = condition_block.split_before(first_op)
        last_body_block = forOp.body.last_block
        assert last_body_block is not None
        rewriter.inline_region_before(forOp.body, end_block)
        iv = condition_block.args[0]

        terminator = last_body_block.last_op
        assert terminator is not None
        assert terminator.has_trait(IsTerminator)

        stepped = Addi(iv, forOp.step)
        rewriter.insert_op(stepped, InsertPoint.before(terminator))

        rewriter.replace_op(
            terminator, Branch(condition_block, stepped, *terminator.operands)
        )

        rewriter.insert_op(
            Branch(condition_block, forOp.lb, *forOp.iter_args),
            InsertPoint.at_end(init_block),
        )

        comparison = Cmpi(iv, forOp.ub, "slt")
        rewriter.insert_op(comparison, InsertPoint.at_end(condition_block))
        cond_branch_op = ConditionalBranch(
            comparison, first_body_block, (), end_block, ()
        )
        rewriter.insert_op(cond_branch_op, InsertPoint.at_end(condition_block))

        rewriter.replace_matched_op([], condition_block.args[1:])


class ConvertScfToCf(ModulePass):
    """ """

    name = "convert-scf-to-cf"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(GreedyRewritePatternApplier([ForLowering()])).rewrite_op(
            op
        )
