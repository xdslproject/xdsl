from xdsl.dialects.builtin import ModuleOp, UnrealizedConversionCastOp
from xdsl.ir.core import Attribute, MLContext, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class ReconcileUnrealizedCasts(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: UnrealizedConversionCastOp, rewriter: PatternRewriter
    ):
        """
        Simplify or remove the cast if possible.

        For each cast:

        1. If the input and output types match, remove the cast entirely.
        2. If a use of the cast is another cast, then pass the input to that cast.
            - If there are no uses other than casts, remove the cast in this operation.

        If each cast was removed, delete the entire operation.
        """

        new_inputs: list[SSAValue] = []
        new_result_types: list[Attribute] = []
        old_indices: list[int] = []

        for i, (o, r) in enumerate(zip(op.operands, op.results)):
            if o.type == r.type:
                r.replace_by(o)
                continue

            for use in tuple(r.uses):
                if isinstance(use.operation, UnrealizedConversionCastOp):
                    use.operation.operands[use.index] = o
                else:
                    # There is a use of the result that is not a cast.
                    new_inputs.append(o)
                    new_result_types.append(r.type)
                    old_indices.append(i)

        if not new_inputs:
            # Replaced all uses, no need for new casts.
            rewriter.erase_matched_op()
            return

        if len(new_inputs) == len(op.operands):
            # There are as many new casts as there were old casts, just keep this one.
            return

        # Some of the cast values are not used, remove them by replacing this op
        new_cast = UnrealizedConversionCastOp.get(new_inputs, new_result_types)
        rewriter.insert_op_before_matched_op(new_cast)

        for new_index, old_index in enumerate(old_indices):
            op.results[old_index].replace_by(new_cast.results[new_index])

        rewriter.erase_matched_op()


class ReconcileUnrealizedCastsPass(ModulePass):
    name = "reconcile-unrealized-casts"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(ReconcileUnrealizedCasts()).rewrite_module(op)
