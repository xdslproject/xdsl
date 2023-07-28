from xdsl.dialects import func, riscv_func
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LowerFuncOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        name = op.sym_name.data

        # TODO: add support for user defined functions
        assert name == "main", "Only support lowering main function for now"
        assert not op.body.blocks[
            0
        ].args, "only support functions with no arguments for now"

        region = op.regions[0]

        new_func = riscv_func.FuncOp(
            op.sym_name.data, rewriter.move_region_contents_to_new_regions(region)
        )

        rewriter.replace_matched_op(new_func)


class LowerReturnOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.Return, rewriter: PatternRewriter):
        # TODO: add support for optional argument
        assert not op.arguments, "Only support return with no arguments for now"

        rewriter.replace_matched_op(riscv_func.ReturnOp(()))


class LowerFuncToRiscvFunc(ModulePass):
    name = "lower-func-to-riscv-func"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerFuncOp()).rewrite_module(op)
        PatternRewriteWalker(LowerReturnOp()).rewrite_module(op)
