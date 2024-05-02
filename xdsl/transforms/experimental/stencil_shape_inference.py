from collections.abc import Iterable
from functools import reduce
from typing import TypeVar, cast

from xdsl.dialects import builtin
from xdsl.dialects.stencil import (
    AccessOp,
    ApplyOp,
    BufferOp,
    CombineOp,
    DynAccessOp,
    FieldType,
    IndexAttr,
    LoadOp,
    StencilBoundsAttr,
    StoreOp,
    TempType,
)
from xdsl.ir import Attribute, BlockArgument, MLContext, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.experimental.convert_stencil_to_ll_mlir import assert_subset
from xdsl.utils.hints import isa

_OpT = TypeVar("_OpT", bound=Operation)


def all_matching_uses(
    op_res: Iterable[SSAValue], op_type: type[_OpT]
) -> Iterable[_OpT]:
    for res in op_res:
        for use in res.uses:
            if isinstance(use.operation, op_type):
                yield use.operation


def infer_core_size(op: LoadOp) -> tuple[IndexAttr, IndexAttr]:
    """
    This method infers the core size (as used in DimsHelper)
    from an LoadOp by walking the def-use chain down to the `apply`
    """
    applies: list[ApplyOp] = list(all_matching_uses([op.res], ApplyOp))
    assert len(applies) > 0, "Load must be followed by Apply!"

    shape_lb: None | IndexAttr = None
    shape_ub: None | IndexAttr = None

    for apply in applies:
        # assert apply.lb is not None and apply.ub is not None
        assert apply.res
        res_type = cast(TempType[Attribute], apply.res[0])
        assert isinstance(res_type.bounds, StencilBoundsAttr)
        shape_lb = IndexAttr.min(res_type.bounds.lb, shape_lb)
        shape_ub = IndexAttr.max(res_type.bounds.ub, shape_ub)

    assert shape_lb is not None
    assert shape_ub is not None
    return shape_lb, shape_ub


class CombineOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CombineOp, rewriter: PatternRewriter, /):
        # Get each result group
        combined_res = op.results_[0 : len(op.lower)]
        lowerext_res = op.results_[len(op.lower) : len(op.lower) + len(op.lowerext)]
        upperext_res = op.results_[len(op.lower) + len(op.lowerext) :]

        combined_bounds = [
            cast(TempType[Attribute], r.type).bounds for r in combined_res
        ]
        lowerext_bounds = [
            cast(TempType[Attribute], r.type).bounds for r in lowerext_res
        ]
        upperext_bounds = [
            cast(TempType[Attribute], r.type).bounds for r in upperext_res
        ]
        assert isa(combined_bounds, list[StencilBoundsAttr])
        assert isa(lowerext_bounds, list[StencilBoundsAttr])
        assert isa(upperext_bounds, list[StencilBoundsAttr])

        lower_bounds = list[StencilBoundsAttr]()
        upper_bounds = list[StencilBoundsAttr]()

        for c in combined_bounds:
            newub = list(c.ub)
            newub[op.dim.value.data] = op.index.value.data
            newl = StencilBoundsAttr.new((c.lb, IndexAttr.get(*newub)))
            lower_bounds.append(newl)

            newlb = list(c.lb)
            newlb[op.dim.value.data] = op.index.value.data
            newu = StencilBoundsAttr.new((IndexAttr.get(*newlb), c.ub))
            upper_bounds.append(newu)

        # Handle combined lower results
        for b, l in zip(lower_bounds, op.lower, strict=True):
            assert isa(l.type, TempType[Attribute])
            l.type = TempType(l.type.bounds | b, l.type.element_type)

        # Handle combined upper results
        for b, u in zip(upper_bounds, op.upper, strict=True):
            assert isa(u.type, TempType[Attribute])
            u.type = TempType(u.type.bounds | b, u.type.element_type)

        # Handle lowerext results
        for r, o in zip(lowerext_bounds, op.lowerext, strict=True):
            assert isa(o.type, TempType[Attribute])
            newub = list(r.ub)
            newub[op.dim.value.data] = op.index.value.data
            newl = StencilBoundsAttr.new((r.lb, IndexAttr.get(*newub)))
            o.type = TempType(o.type.bounds | newl, o.type.element_type)

        # Handle upperext results
        for r, o in zip(upperext_bounds, op.upperext, strict=True):
            assert isa(o.type, TempType[Attribute])
            newlb = list(r.lb)
            newlb[op.dim.value.data] = op.index.value.data
            newu = StencilBoundsAttr.new((IndexAttr.get(*newlb), r.ub))
            o.type = TempType(o.type.bounds | newu, o.type.element_type)


class LoadOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        field = op.field.type
        assert isa(field, FieldType[Attribute])
        temp = op.res.type
        assert isa(temp, TempType[Attribute])

        assert_subset(field, temp)


class StoreOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter, /):
        temp = op.temp.type
        assert isa(temp, TempType[Attribute])

        op.temp.type = TempType(op.bounds | temp.bounds, temp.element_type)


class AccessOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter):
        apply = op.get_apply()
        assert isa(op.temp.type, TempType[Attribute])
        assert isa(apply.res[0].type, TempType[Attribute])

        temp_type = op.temp.type

        output_size = apply.res[0].type.bounds
        assert isinstance(output_size, StencilBoundsAttr)

        ntype = TempType(
            temp_type.bounds | output_size + op.offset, temp_type.element_type
        )

        op.temp.type = ntype

        assert isinstance(op.temp, BlockArgument)
        assert op.temp.owner.parent_op() is apply
        apply.operands[op.temp.index].type = ntype


class DynAccessOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynAccessOp, rewriter: PatternRewriter):
        apply = op.parent_op()
        assert isinstance(apply, ApplyOp)
        assert isa(op.temp.type, TempType[Attribute])
        assert isa(apply.res[0].type, TempType[Attribute]), f"{apply.res[0]}"

        temp_type = op.temp.type
        output_size = apply.res[0].type.bounds
        assert isinstance(output_size, StencilBoundsAttr)
        ntype = TempType(
            temp_type.bounds | output_size + op.lb | output_size + op.ub,
            temp_type.element_type,
        )

        op.temp.type = ntype

        assert isinstance(op.temp, BlockArgument)
        assert op.temp.owner.parent_op() is apply
        apply.operands[op.temp.index].type = ntype


class ApplyOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        results_bounds = tuple(
            cast(TempType[Attribute], res.type).bounds for res in op.res
        )
        assert isa(results_bounds, tuple[StencilBoundsAttr, ...])
        output_bounds = reduce(lambda l, r: l | r, results_bounds)
        for res in op.res:
            res.type = TempType(
                output_bounds, cast(TempType[Attribute], res.type).element_type
            )


class BufferOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BufferOp, rewriter: PatternRewriter):
        op.temp.type = op.res.type


ShapeInference = GreedyRewritePatternApplier(
    [
        AccessOpShapeInference(),
        ApplyOpShapeInference(),
        BufferOpShapeInference(),
        CombineOpShapeInference(),
        DynAccessOpShapeInference(),
        LoadOpShapeInference(),
        StoreOpShapeInference(),
    ]
)


class StencilShapeInferencePass(ModulePass):
    name = "stencil-shape-inference"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        inference_walker = PatternRewriteWalker(
            ShapeInference,
            apply_recursively=False,
            walk_reverse=True,
            walk_regions_first=False,
        )
        inference_walker.rewrite_module(op)
