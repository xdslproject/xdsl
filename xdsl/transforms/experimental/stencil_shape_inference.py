from collections.abc import Iterable
from functools import reduce
from typing import TypeVar, cast

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
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
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


def update_result_size(
    value: SSAValue, size: StencilBoundsAttr, rewriter: PatternRewriter
):
    """
    Wrapper for corner-case result size updating.
    On the general case, just update the result's type.
    If it is a stencil.apply's result though, it updates all the results of the aplly.
    For each other result updated, it also updates any buffer operation that uses it,
    otherwise the buffer's result might not match anymore.
    """
    if isinstance(value.owner, ApplyOp):
        apply = value.owner
        res_types = (cast(TempType[Attribute], r.type) for r in apply.res)
        newsize = reduce(
            lambda l, r: l | r,
            (
                size,
                *(
                    t.bounds
                    for t in res_types
                    if isinstance(t.bounds, StencilBoundsAttr)
                ),
            ),
        )
        for res in apply.res:
            newtype = TempType(
                newsize, cast(TempType[Attribute], res.type).element_type
            )
            if newtype != res.type:
                rewriter.modify_value_type(res, newtype)
            for use in res.uses:
                if isinstance(use.operation, BufferOp):
                    update_result_size(use.operation.res, newsize, rewriter)
    newsize = size | cast(TempType[Attribute], value.type).bounds
    newtype = TempType(newsize, cast(TempType[Attribute], value.type).element_type)
    if newtype != value.type:
        rewriter.modify_value_type(value, newtype)


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

        lower_bounds = list[StencilBoundsAttr | None]()
        upper_bounds = list[StencilBoundsAttr | None]()

        for c in combined_bounds:
            if not isinstance(c, StencilBoundsAttr):
                lower_bounds.append(None)
                upper_bounds.append(None)
                continue
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
            if b is None:
                continue
            assert isa(l.type, TempType[Attribute])
            update_result_size(l, l.type.bounds | b, rewriter)

        # Handle combined upper results
        for b, u in zip(upper_bounds, op.upper, strict=True):
            if b is None:
                continue
            assert isa(u.type, TempType[Attribute])
            update_result_size(u, u.type.bounds | b, rewriter)

        # Handle lowerext results
        for r, o in zip(lowerext_bounds, op.lowerext, strict=True):
            if not isinstance(r, StencilBoundsAttr):
                continue
            assert isa(o.type, TempType[Attribute])
            newub = list(r.ub)
            newub[op.dim.value.data] = op.index.value.data
            newl = StencilBoundsAttr.new((r.lb, IndexAttr.get(*newub)))
            update_result_size(o, o.type.bounds | newl, rewriter)

        # Handle upperext results
        for r, o in zip(upperext_bounds, op.upperext, strict=True):
            if not isinstance(r, StencilBoundsAttr):
                continue
            assert isa(o.type, TempType[Attribute])
            newlb = list(r.lb)
            newlb[op.dim.value.data] = op.index.value.data
            newu = StencilBoundsAttr.new((IndexAttr.get(*newlb), r.ub))
            update_result_size(o, o.type.bounds | newu, rewriter)


class LoadOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        field = op.field.type
        assert isa(field, FieldType[Attribute])
        temp = op.res.type
        assert isa(temp, TempType[Attribute])


class StoreOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter, /):
        temp = op.temp.type
        assert isa(temp, TempType[Attribute])

        update_result_size(op.temp, temp.bounds | op.bounds, rewriter)


class AccessOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter):
        apply = op.get_apply()
        assert isa(op.temp.type, TempType[Attribute])
        assert isa(apply.res[0].type, TempType[Attribute])

        temp_type = op.temp.type

        output_size = apply.res[0].type.bounds
        if not isinstance(output_size, StencilBoundsAttr):
            return

        update_result_size(
            op.temp, temp_type.bounds | output_size + op.offset, rewriter
        )


class DynAccessOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynAccessOp, rewriter: PatternRewriter):
        apply = op.parent_op()
        assert isinstance(apply, ApplyOp)
        assert isa(op.temp.type, TempType[Attribute])
        assert isa(apply.res[0].type, TempType[Attribute]), f"{apply.res[0]}"

        temp_type = op.temp.type
        output_size = apply.res[0].type.bounds
        if not isinstance(output_size, StencilBoundsAttr):
            return

        update_result_size(
            op.temp,
            temp_type.bounds | output_size + op.lb | output_size + op.ub,
            rewriter,
        )


class ApplyOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        for i, arg in enumerate(op.region.block.args):
            if isa(arg.type, TempType[Attribute]) and isinstance(
                arg.type.bounds, StencilBoundsAttr
            ):
                assert isa(ot := op.operands[i].type, TempType[Attribute])
                new_bounds = arg.type.bounds | ot.bounds
                if new_bounds != ot.bounds:
                    update_result_size(op.operands[i], new_bounds, rewriter)
                if new_bounds != arg.type.bounds:
                    update_result_size(arg, new_bounds, rewriter)


class BufferOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BufferOp, rewriter: PatternRewriter):
        res_bounds = cast(TempType[Attribute], op.res.type).bounds
        if not isinstance(res_bounds, StencilBoundsAttr):
            return
        op.temp.type = op.res.type
        update_result_size(op.temp, res_bounds, rewriter)
