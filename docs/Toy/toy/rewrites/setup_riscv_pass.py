from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir.core import Block, MLContext, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


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
        text_section = riscv.AssemblySectionOp(
            ".text", rewriter.move_region_contents_to_new_regions(op.regions[0])
        )

        op.body.add_block(Block([heap_section, data_section, text_section]))


class SetupRiscvPass(ModulePass):
    name = "setup-lowering-to-riscv"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(AddSections()).rewrite_module(op)
