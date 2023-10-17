from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, MLContext, Region
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

        rewriter.insert_op_at_start((heap_section, data_section), op.body.block)


class SetupRiscvPass(ModulePass):
    name = "setup-lowering-to-riscv"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(AddSections()).rewrite_module(op)
