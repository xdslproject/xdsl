from xdsl.context import MLContext
from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


class AddSections(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ModuleOp, rewriter: PatternRewriter):
        # bss stands for block starting symbol
        heap_section = riscv.AssemblySectionOp(
            ".bss",
            Region(
                Block(
                    [
                        riscv.LabelOp("heap"),
                        riscv.DirectiveOp(".space", f"{1024}"),  # 1kb
                    ]
                )
            ),
        )
        data_section = riscv.AssemblySectionOp(".data", Region(Block()))

        rewriter.insert_op(
            (heap_section, data_section), InsertPoint.at_start(op.body.block)
        )


class SetupRiscvPass(ModulePass):
    name = "setup-lowering-to-riscv"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(AddSections(), apply_recursively=False).rewrite_module(op)
