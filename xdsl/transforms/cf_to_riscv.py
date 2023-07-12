import random
from abc import ABC

from xdsl.dialects import cf, riscv
from xdsl.dialects.builtin import ModuleOp, UnrealizedConversionCastOp
from xdsl.ir import Block, MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def label_block(
    block: Block, rewriter: PatternRewriter, at_start: bool = True
) -> riscv.LabelOp:
    if at_start and isinstance(block.ops.first, riscv.LabelOp):
        return block.ops.first
    elif not at_start and isinstance(block.ops.last, riscv.LabelOp):
        return block.ops.last
    else:
        parent_region = block.parent
        block_idx = (
            parent_region.get_block_index(block)
            if parent_region
            else random.randint(0, 100)
        )
        label_str = "bb" + str(block_idx)
        if not at_start:
            label_str += "end"

        label_op = riscv.LabelOp(label_str)
        if parent_region:
            parent_region.detach_block(block)

        if at_start:
            rewriter.insert_op_at_start(label_op, block)
        else:
            rewriter.insert_op_at_end(label_op, block)

        if parent_region:
            parent_region.insert_block(block, block_idx)

        return label_op


def add_end_label(block: Block) -> riscv.LabelOp | None:
    if parent_region := block.parent:
        block_idx = parent_region.get_block_index(block)
        label_str = "bb" + str(block_idx) + "end"
        label_op = riscv.LabelOp(label_str)

        parent_region.insert_block(Block([label_op, riscv.NopOp()]), block_idx + 1)

        return label_op


def add_jump(block: Block, label: riscv.LabelOp):
    if parent_region := block.parent:
        block_idx = parent_region.get_block_index(block)
        parent_region.detach_block(block)

        jmp = riscv.JOp(label.label)
        parent_region.insert_block(Block([jmp]), block_idx)

        parent_region.insert_block(block, block_idx)


class LowerConditionalBranchToRISCV(RewritePattern, ABC):
    """ """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.ConditionalBranch, rewriter: PatternRewriter, /):
        _ = label_block(op.then_block, rewriter)
        else_label = label_block(op.else_block, rewriter)
        else_label_end = add_end_label(op.else_block)
        if else_label_end:
            add_jump(op.then_block, else_label_end)

        cond = UnrealizedConversionCastOp.get(
            [op.cond], [riscv.RegisterType(riscv.Register())]
        )
        zero = riscv.GetRegisterOp(riscv.Registers.ZERO)
        branch = riscv.BeqOp(cond.results[0], zero, else_label.label)

        rewriter.replace_matched_op([cond, zero, branch])


class CfToRISCV(ModulePass):
    """ """

    name = "cf-to-riscv"

    # lower to func.call
    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerConditionalBranchToRISCV()])
        ).rewrite_module(op)
