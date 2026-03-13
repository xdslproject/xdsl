from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import linalg
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class GeneralizeNamedOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.NamedOperation, rewriter: PatternRewriter
    ) -> None:
        """
        Rewrite a named linalg op to `linalg.generic`.
        """

        indexing_maps = tuple(op.get_indexing_maps())

        generic = linalg.GenericOp(
            op.inputs,
            op.outputs,
            op.body.clone(),
            ArrayAttr(AffineMapAttr(map_) for map_ in indexing_maps),
            op.get_iterator_types(),
            [res.type for res in op.res],
        )
        rewriter.replace_op(op, generic, new_results=generic.results)


@dataclass(frozen=True)
class LinalgGeneralizeNamedOpsPass(ModulePass):
    """
    Converts linalg named ops to linalg.generic.
    """

    name = "linalg-generalize-named-ops"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([GeneralizeNamedOpPattern()]),
            apply_recursively=False,
        ).rewrite_module(op)
