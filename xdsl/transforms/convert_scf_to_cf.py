from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.arith import Addi, Cmpi
from xdsl.dialects.cf import Branch, ConditionalBranch
from xdsl.dialects.scf import For, If
from xdsl.ir import Block
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


class IfLowering(RewritePattern):
    """Lowers `scf.if` to conditional branching."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, ifOp: If, rewriter: PatternRewriter, /):
        condition_block = ifOp.parent_block()
        assert condition_block is not None

        remaining_ops_block = condition_block.split_before(ifOp)
        if ifOp.results:
            parent = condition_block.parent_region()
            assert parent is not None
            continue_block = Block(arg_types=ifOp.result_types)
            parent.insert_block_before(continue_block, remaining_ops_block)
            rewriter.insert_op(
                Branch(remaining_ops_block), InsertPoint.at_end(continue_block)
            )
        else:
            continue_block = remaining_ops_block

        then_region = ifOp.true_region
        then_block = then_region.first_block
        assert then_block is not None
        assert then_region.last_block is not None
        then_terminator = then_region.last_block.last_op
        assert then_terminator is not None
        then_terminator_operands = then_terminator.operands
        rewriter.insert_op(
            Branch(continue_block, *then_terminator_operands),
            InsertPoint.at_end(then_region.last_block),
        )

        rewriter.erase_op(then_terminator)
        rewriter.inline_region_before(then_region, continue_block)

        else_region = ifOp.false_region
        else_block = else_region.first_block
        assert else_block is not None
        assert else_region.last_block is not None
        else_terminator = else_region.last_block.last_op
        assert else_terminator is not None
        else_terminator_operands = else_terminator.operands
        rewriter.insert_op(
            Branch(continue_block, *else_terminator_operands),
            InsertPoint.at_end(else_region.last_block),
        )

        rewriter.erase_op(else_terminator)
        rewriter.inline_region_before(else_region, continue_block)

        rewriter.insert_op(
            ConditionalBranch(ifOp.cond, then_block, (), else_block, ()),
            InsertPoint.at_end(condition_block),
        )

        rewriter.replace_matched_op([], continue_block.args)


class ForLowering(RewritePattern):
    """Lowers `scf.for` to conditional branching."""

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
