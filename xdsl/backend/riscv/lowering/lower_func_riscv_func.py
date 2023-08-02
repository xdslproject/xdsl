from xdsl.dialects import func, riscv_func
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LowerFuncOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        name = op.sym_name.data

        if name != "main":
            raise NotImplementedError("Only support lowering main function for now")

        if op.body.blocks[0].args:
            raise NotImplementedError(
                "Only support functions with no arguments for now"
            )

        new_func = riscv_func.FuncOp(
            op.sym_name.data, rewriter.move_region_contents_to_new_regions(op.body)
        )

        rewriter.replace_matched_op(new_func)


class LowerFuncCallOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.Call, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("Function call lowering not implemented yet")


class LowerReturnOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.Return, rewriter: PatternRewriter):
        if op.arguments:
            raise NotImplementedError("Only support return with no arguments for now")

        rewriter.replace_matched_op(riscv_func.ReturnOp(()))


class LowerFuncToRiscvFunc(ModulePass):
    name = "lower-func-to-riscv-func"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerFuncOp(),
                    LowerFuncCallOp(),
                    LowerReturnOp(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
