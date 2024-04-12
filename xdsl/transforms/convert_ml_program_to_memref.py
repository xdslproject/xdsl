from xdsl.dialects import memref, ml_program
from xdsl.dialects.builtin import ModuleOp, UnitAttr, UnrealizedConversionCastOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class ConvertGlobalPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: ml_program.Global, rewriter: PatternRewriter
    ) -> None:
        if op.value is None:
            raise NotImplementedError(
                "Converting ml_program.global with no value not implemented"
            )
        rewriter.replace_matched_op(
            (
                memref.Global.get(
                    op.sym_name,
                    op.type,
                    op.value,
                    op.sym_visibility,
                    UnitAttr() if op.is_mutable is None else None,
                ),
            )
        )


class ConvertGlobalLoadConst(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: ml_program.GlobalLoadConstant, rewriter: PatternRewriter
    ) -> None:
        rewriter.replace_matched_op(
            (
                mem := memref.GetGlobal.get(op.global_, op.result.type),
                UnrealizedConversionCastOp.get(mem.results, (op.result.type,)),
            )
        )


class ConvertMlProgramToMemrefPass(ModulePass):

    name = "convert-ml-program-to-memref"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [ConvertGlobalPattern(), ConvertGlobalLoadConst()]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
