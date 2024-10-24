from typing import cast

from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.arith import Addi, Cmpi, IndexCastOp
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, i32
from xdsl.dialects.cf import Branch, ConditionalBranch, Switch
from xdsl.dialects.scf import For, If, IndexSwitchOp, Yield
from xdsl.ir import Block, Region
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
    """
    Lowers `scf.if` to conditional branching.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, if_op: If, rewriter: PatternRewriter, /):
        condition_block = if_op.parent_block()
        assert condition_block is not None

        # Start by splitting the block containing the 'scf.if' into two parts.
        # The part before will contain the condition, the part after will be the
        # continuation point.
        remaining_ops_block = condition_block.split_before(if_op)
        if if_op.results:
            parent = condition_block.parent_region()
            assert parent is not None
            continue_block = Block(arg_types=if_op.result_types)
            parent.insert_block_before(continue_block, remaining_ops_block)
            rewriter.insert_op(
                Branch(remaining_ops_block), InsertPoint.at_end(continue_block)
            )
        else:
            continue_block = remaining_ops_block

        # Move blocks from the "then" region to the region containing 'scf.if',
        # place it before the continuation block, and branch to it.
        then_region = if_op.true_region
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

        # Move blocks from the "else" region (if present) to the region containing
        # 'scf.if', place it before the continuation block and branch to it.  It
        # will be placed after the "then" regions.
        else_region = if_op.false_region
        else_block = else_region.first_block
        if else_block is not None:
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
        else:
            else_block = continue_block

        # Branch to either the then_block or else_block
        rewriter.insert_op(
            ConditionalBranch(if_op.cond, then_block, (), else_block, ()),
            InsertPoint.at_end(condition_block),
        )

        # Remove the original `scf.if` operation
        rewriter.replace_matched_op([], continue_block.args)


class ForLowering(RewritePattern):
    """Lowers `scf.for` to conditional branching."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, for_op: For, rewriter: PatternRewriter):
        # Start by splitting the block containing the 'scf.for' into two parts.
        # The part before will get the init code, the part after will be the end
        # point.
        init_block = for_op.parent_block()
        if init_block is None:
            return
        end_block = init_block.split_before(for_op)

        # Use the first block of the loop body as the condition block since it is the
        # block that has the induction variable and loop-carried values as arguments.
        # Split out all operations from the first block into a new block. Move all
        # body blocks from the loop body region to the region containing the loop.
        condition_block = for_op.body.first_block
        assert condition_block is not None
        first_op = condition_block.first_op
        assert first_op is not None
        first_body_block = condition_block.split_before(first_op)
        last_body_block = for_op.body.last_block
        assert last_body_block is not None
        rewriter.inline_region_before(for_op.body, end_block)
        iv = condition_block.args[0]

        # Append the induction variable stepping logic to the last body block and
        # branch back to the condition block. Loop-carried values are taken from
        # operands of the loop terminator.
        terminator = last_body_block.last_op
        assert terminator is not None
        assert terminator.has_trait(IsTerminator)

        stepped = Addi(iv, for_op.step)
        rewriter.insert_op(stepped, InsertPoint.before(terminator))

        rewriter.replace_op(
            terminator, Branch(condition_block, stepped, *terminator.operands)
        )

        # The initial values of loop-carried values are obtained from the operands
        # of the loop operation.
        rewriter.insert_op(
            Branch(condition_block, for_op.lb, *for_op.iter_args),
            InsertPoint.at_end(init_block),
        )

        # With the body block done, we can fill in the condition block.
        comparison = Cmpi(iv, for_op.ub, "slt")
        rewriter.insert_op(comparison, InsertPoint.at_end(condition_block))
        cond_branch_op = ConditionalBranch(
            comparison, first_body_block, (), end_block, ()
        )
        rewriter.insert_op(cond_branch_op, InsertPoint.at_end(condition_block))

        # The result of the loop operation are the values of the condition block
        # arguments except the induction variable on the last iteration.
        rewriter.replace_matched_op([], condition_block.args[1:])


class SwitchLowering(RewritePattern):
    """Lowers `scf.index_switch` to `cf.switch`."""

    @staticmethod
    def _convert_region(
        region: Region, continue_block: Block, rewriter: PatternRewriter
    ) -> Block:
        block = region.first_block
        assert block is not None

        # Convert yield op to a branch to the continue block
        yield_op = block.last_op
        assert isinstance(yield_op, Yield)
        rewriter.replace_op(yield_op, Branch(continue_block, *yield_op.operands))

        # Inline the region
        rewriter.inline_region_before(region, continue_block)
        return block

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: IndexSwitchOp, rewriter: PatternRewriter):
        # Split the block at `op`
        condition_block = op.parent_block()
        if condition_block is None:
            return

        continue_block = condition_block.split_before(op)

        # Create the arguments on the continue block with which to replace the
        # results of the op.
        for i, ty in enumerate(op.result_types):
            rewriter.insert_block_argument(continue_block, i, ty)

        # Convert the case regions
        case_successors = tuple(
            self._convert_region(region, continue_block, rewriter)
            for region in op.case_regions
        )
        case_values: tuple[int, ...] = tuple(
            cast(int, i.data) for i in op.cases.data.data
        )

        # Convert the default region
        default_block = self._convert_region(
            op.default_region, continue_block, rewriter
        )

        # Cast switch index to integer case value
        case_value = IndexCastOp(op.arg, i32)
        rewriter.insert_op(case_value, InsertPoint.at_end(condition_block))

        # Create the switch
        case_operands = tuple(() for _ in case_successors)
        rewriter.insert_op(
            Switch(
                case_value,
                default_block,
                (),
                DenseIntOrFPElementsAttr.vector_from_list(case_values, i32),
                case_successors,
                case_operands,
            ),
            InsertPoint.at_end(condition_block),
        )

        rewriter.replace_matched_op((), continue_block.args)


class ConvertScfToCf(ModulePass):
    """
    Lower `scf.for` and `scf.if` to unstructured control flow.
    Implementations are direct translations of the mlir versions found at
    https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp
    """

    name = "convert-scf-to-cf"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    SwitchLowering(),
                    IfLowering(),
                    ForLowering(),
                ]
            )
        ).rewrite_op(op)
