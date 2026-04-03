from xdsl.context import Context
from xdsl.dialects import builtin, eqsat_pdl_interp, pdl_interp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class ConvertGetResultOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: pdl_interp.GetResultOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_matched_op(eqsat_pdl_interp.GetResultOp(op.index, op.input_op))


class ConvertGetResultsOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: pdl_interp.GetResultsOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_matched_op(
            eqsat_pdl_interp.GetResultsOp(op.index, op.input_op, op.value.type)
        )


class ConvertGetDefiningOpOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: pdl_interp.GetDefiningOpOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_matched_op(eqsat_pdl_interp.GetDefiningOpOp(op.value))


class ConvertReplaceOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: pdl_interp.ReplaceOp, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(
            eqsat_pdl_interp.ReplaceOp(op.input_op, list(op.repl_values))
        )


class ConvertCreateOperationOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: pdl_interp.CreateOperationOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_matched_op(
            eqsat_pdl_interp.CreateOperationOp(
                op.constraint_name,
                op.inferred_result_types,
                op.input_attribute_names,
                op.input_operands,
                op.input_attributes,
                op.input_result_types,
            )
        )


class ConvertRecordMatchOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: pdl_interp.RecordMatchOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_matched_op(
            eqsat_pdl_interp.RecordMatchOp(
                op.rewriter,
                op.rootKind,
                op.generatedOps,
                op.benefit,
                op.inputs,
                op.matched_ops,
                op.dest,
            )
        )


class ConvertFinalizeOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: pdl_interp.FinalizeOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_matched_op(eqsat_pdl_interp.FinalizeOp())


class ConvertPDLInterpToEqsatPDLInterpPass(ModulePass):
    name = "convert-pdl-interp-to-eqsat-pdl-interp"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertGetResultOp(),
                    ConvertGetResultsOp(),
                    ConvertGetDefiningOpOp(),
                    ConvertReplaceOp(),
                    ConvertCreateOperationOp(),
                    ConvertRecordMatchOp(),
                    ConvertFinalizeOp(),
                ]
            )
        ).rewrite_module(op)
