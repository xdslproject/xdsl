from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import linalg
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, ModuleOp
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


def get_iterator_types(
    op: linalg.NamedOperation,
    indexing_maps: tuple[AffineMap, ...],
    num_inputs: int,
) -> ArrayAttr[linalg.IteratorTypeAttr]:
    """
    Return iterator types for the generalized op.

    Reuse iterator types when present; otherwise infer them from the
    output indexing maps.
    """
    existing_iterator_types = op.attributes.get("iterator_types")
    if isa(existing_iterator_types, ArrayAttr[linalg.IteratorTypeAttr]):
        return existing_iterator_types

    output_maps = indexing_maps[num_inputs:]
    num_loops = indexing_maps[0].num_dims
    parallel_dims: set[int] = set()
    for output_map in output_maps:
        for result_expr in output_map.results:
            parallel_dims.update(result_expr.used_dims())

    inferred_iterator_types: list[linalg.IteratorTypeAttr] = []
    for dim in range(num_loops):
        if dim in parallel_dims:
            inferred_iterator_types.append(linalg.IteratorTypeAttr.parallel())
        else:
            inferred_iterator_types.append(linalg.IteratorTypeAttr.reduction())

    return ArrayAttr(inferred_iterator_types)


def generalize_named_op_precondition(op: linalg.NamedOperation) -> bool:
    """
    Check whether a named op can be generalized.
    """
    if isinstance(op, linalg.GenericOp):
        return False
    if len(op.regions) != 1:
        return False
    try:
        tuple(op.get_indexing_maps())
    except NotImplementedError:
        return False
    return True


def generalize_named_op(
    rewriter: PatternRewriter, op: linalg.NamedOperation
) -> linalg.GenericOp | None:
    """
    Rewrite a named linalg op to `linalg.generic`.
    """
    if not generalize_named_op_precondition(op):
        return None

    indexing_maps = tuple(op.get_indexing_maps())

    generic = linalg.GenericOp(
        op.inputs,
        op.outputs,
        op.body.clone(),
        ArrayAttr(AffineMapAttr(map_) for map_ in indexing_maps),
        get_iterator_types(op, indexing_maps, len(op.inputs)),
        [res.type for res in op.res],
    )
    rewriter.replace_op(op, generic, new_results=generic.results)
    return generic


class GeneralizeNamedOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.NamedOperation, rewriter: PatternRewriter
    ) -> None:
        generalize_named_op(rewriter, op)


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
