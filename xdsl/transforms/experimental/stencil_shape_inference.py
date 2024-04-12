from collections.abc import Iterable
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

        # Handle combined lower results
        for c, l in zip(combined_res, op.lower, strict=True):
            c_type = cast(TempType[Attribute], c.type)
            assert isinstance(c_type.bounds, StencilBoundsAttr)
            # Get the inferred bounds on the combined result
            c_bounds = c_type.bounds
            assert isa(l.type, TempType[Attribute])

            # Recover existing bounds on the lower and upper input if any
            lb = None
            ub = None
            if isinstance(l.type.bounds, StencilBoundsAttr):
                lb = l.type.bounds.lb
                ub = l.type.bounds.ub

            # Compute the new extreme bounds as usual.
            lb = IndexAttr.min(c_bounds.lb, lb)
            # Compute the combine bounds
            c_bound_c = list(c_bounds.ub)
            c_bound_c[op.dim.value.data] = op.index.value.data
            c_bound = IndexAttr.get(*c_bound_c)
            ub = IndexAttr.max(c_bound, ub)
            bounds = StencilBoundsAttr(zip(lb, ub))
            l.type = TempType(bounds, l.type.element_type)

        # Handle combined upper results
        for c, u in zip(combined_res, op.upper, strict=True):
            c_type = cast(TempType[Attribute], c.type)
            assert isinstance(c_type.bounds, StencilBoundsAttr)
            # Get the inferred bounds on the combined result
            c_bounds = c_type.bounds
            assert isa(u.type, TempType[Attribute])

            # Recover existing bounds on the lower and upper input if any
            lb = None
            ub = None
            if isinstance(u.type.bounds, StencilBoundsAttr):
                lb = u.type.bounds.lb
                ub = u.type.bounds.ub

            # Compute the new extreme bounds as usual.
            ub = IndexAttr.max(c_bounds.ub, ub)
            # Compute the combine bounds
            c_bound_c = list(c_bounds.lb)
            c_bound_c[op.dim.value.data] = op.index.value.data
            c_bound = IndexAttr.get(*c_bound_c)
            lb = IndexAttr.min(c_bound, lb)
            bounds = StencilBoundsAttr(zip(lb, ub))
            u.type = TempType(bounds, u.type.element_type)

        # Handle lowerext results
        for r, o in zip(lowerext_res, op.lowerext, strict=True):
            assert isa(o.type, TempType[Attribute])
            assert isa(r.type, TempType[Attribute])
            r_bounds = r.type.bounds
            assert isinstance(r_bounds, StencilBoundsAttr)
            # Recover existing bounds on the upperext input if any
            lb = None
            ub = None
            if isinstance(o.type.bounds, StencilBoundsAttr):
                lb = o.type.bounds.lb
                ub = o.type.bounds.ub

            ub_c = list(r_bounds.ub)
            ub_c[op.dim.value.data] = op.index.value.data

            ub_c = IndexAttr.get(*ub_c)

            lb = IndexAttr.min(r_bounds.lb, lb)
            ub = IndexAttr.max(ub_c, ub)

            o.type = TempType(StencilBoundsAttr(zip(lb, ub)), o.type.element_type)

        # Handle upperext results
        for r, o in zip(upperext_res, op.upperext, strict=True):
            assert isa(o.type, TempType[Attribute])
            assert isa(r.type, TempType[Attribute])
            r_bounds = r.type.bounds
            assert isinstance(r_bounds, StencilBoundsAttr)
            # Recover existing bounds on the upperext input if any
            lb = None
            ub = None
            if isinstance(o.type.bounds, StencilBoundsAttr):
                lb = o.type.bounds.lb
                ub = o.type.bounds.ub

            lb_c = list(r_bounds.lb)
            lb_c[op.dim.value.data] = op.index.value.data

            lb_c = IndexAttr.get(*lb_c)

            lb = IndexAttr.min(lb_c, lb)
            ub = IndexAttr.max(r_bounds.ub, ub)

            o.type = TempType(StencilBoundsAttr(zip(lb, ub)), o.type.element_type)


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
        temp_lb = None
        temp_ub = None
        if isinstance(temp.bounds, StencilBoundsAttr):
            temp_lb = temp.bounds.lb
            temp_ub = temp.bounds.ub

        temp_lb = IndexAttr.min(op.lb, temp_lb)
        temp_ub = IndexAttr.max(op.ub, temp_ub)
        op.temp.type = TempType(tuple(zip(temp_lb, temp_ub)), temp.element_type)


class AccessOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter):
        apply = op.parent_op()
        assert isinstance(apply, ApplyOp)
        assert isa(op.temp.type, TempType[Attribute])
        assert isinstance(op.temp, BlockArgument)
        assert op.temp.block.parent_op() is apply
        assert isa(apply.res[0].type, TempType[Attribute]), f"{apply.res[0]}"

        temp_type = op.temp.type
        temp_lb = None
        temp_ub = None
        if isinstance(temp_type.bounds, StencilBoundsAttr):
            temp_lb = temp_type.bounds.lb
            temp_ub = temp_type.bounds.ub
        output_size = apply.res[0].type.bounds
        assert isinstance(output_size, StencilBoundsAttr)

        lb = IndexAttr.min(output_size.lb + op.offset, temp_lb)
        ub = IndexAttr.max(output_size.ub + op.offset, temp_ub)
        ntype = TempType(tuple(zip(lb, ub)), temp_type.element_type)

        op.temp.type = ntype


class DynAccessOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynAccessOp, rewriter: PatternRewriter):
        apply = op.parent_op()
        assert isinstance(apply, ApplyOp)
        assert isa(op.temp.type, TempType[Attribute])
        assert isinstance(op.temp, BlockArgument)
        assert op.temp.block.parent_op() is apply
        assert isa(apply.res[0].type, TempType[Attribute]), f"{apply.res[0]}"

        temp_type = op.temp.type
        temp_lb = None
        temp_ub = None
        if isinstance(temp_type.bounds, StencilBoundsAttr):
            temp_lb = temp_type.bounds.lb
            temp_ub = temp_type.bounds.ub
        output_size = apply.res[0].type.bounds
        assert isinstance(output_size, StencilBoundsAttr)

        lb = IndexAttr.min(output_size.lb + op.lb, temp_lb)
        ub = IndexAttr.max(output_size.ub + op.ub, temp_ub)
        ntype = TempType(tuple(zip(lb, ub)), temp_type.element_type)

        op.temp.type = ntype


class ApplyOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        if len(op.res) < 1:
            return
        res_type = op.res[0].type
        assert isa(res_type, TempType[Attribute])
        assert isinstance(res_type.bounds, StencilBoundsAttr)
        ntype = res_type
        assert isinstance(ntype.bounds, StencilBoundsAttr)

        for i, arg in enumerate(op.region.block.args):
            if not isa(arg.type, TempType[Attribute]):
                continue
            op.operands[i].type = arg.type


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
            walk_regions_first=True,
        )
        inference_walker.rewrite_module(op)
