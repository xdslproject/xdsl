from xdsl.context import Context
from xdsl.dialects import linalg, memref_stream
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr, IntegerAttr, ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def iterator_type_attr(t: linalg.IteratorTypeAttr) -> memref_stream.IteratorTypeAttr:
    match t.data:
        case linalg.IteratorType.PARALLEL:
            return memref_stream.IteratorTypeAttr.parallel()
        case linalg.IteratorType.REDUCTION:
            return memref_stream.IteratorTypeAttr.reduction()
        case linalg.IteratorType.WINDOW:
            raise NotImplementedError("Cannot convert window iterator type")


class ConvertGenericOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.GenericOp, rewriter: PatternRewriter
    ) -> None:
        if op.res:
            raise NotImplementedError(
                "converting linalg.generic with results not supported"
            )

        # The memref_stream.generic op may take as arguments memrefs, scalars, or streams,
        # the latter of which does not carry shape information. linalg.generic constructs
        # the nested loop bounds from the shapes of the inputs, so we need to cache that
        # derived information here, as we may not be able to recover it later.
        ubs = op.get_static_loop_ranges()
        index = IndexType()
        bounds = ArrayAttr(IntegerAttr(IntAttr(ub), index) for ub in ubs)

        iterator_types = ArrayAttr(iterator_type_attr(t) for t in op.iterator_types)

        rewriter.replace_op(
            op,
            memref_stream.GenericOp(
                op.inputs,
                op.outputs,
                (),
                rewriter.move_region_contents_to_new_regions(op.body),
                op.indexing_maps,
                iterator_types,
                bounds,
                ArrayAttr(()),
                op.doc,
                op.library_call,
            ),
        )


class ConvertYieldOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.YieldOp, rewriter: PatternRewriter) -> None:
        rewriter.replace_op(op, memref_stream.YieldOp(*op.operands))


class ConvertLinalgToMemRefStreamPass(ModulePass):
    name = "convert-linalg-to-memref-stream"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [ConvertGenericOpPattern(), ConvertYieldOpPattern()]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
