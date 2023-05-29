from typing import Iterable, TypeVar, cast
from xdsl.dialects import builtin
from xdsl.dialects.experimental.stencil import (
    AccessOp,
    ApplyOp,
    FieldType,
    IndexAttr,
    LoadOp,
    StencilBoundsAttr,
    StoreOp,
    TempType,
)

from xdsl.ir import Attribute, MLContext, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.experimental.ConvertStencilToLLMLIR import assert_subset
from xdsl.utils.hints import isa

_OpT = TypeVar("_OpT", bound=Operation)


def all_matching_uses(op_res: Iterable[SSAValue], typ: type[_OpT]) -> Iterable[_OpT]:
    for res in op_res:
        for use in res.uses:
            if isinstance(use.operation, typ):
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
        res_typ = cast(TempType[Attribute], apply.res[0])
        assert isinstance(res_typ.bounds, StencilBoundsAttr)
        shape_lb = IndexAttr.min(res_typ.bounds.lb, shape_lb)
        shape_ub = IndexAttr.max(res_typ.bounds.ub, shape_ub)

    assert shape_lb is not None and shape_ub is not None
    return shape_lb, shape_ub


class LoadOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        field = op.field.typ
        assert isa(field, FieldType[Attribute])
        temp = op.res.typ
        assert isa(temp, TempType[Attribute])

        assert_subset(field, temp)


class StoreOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter, /):
        temp = op.temp.typ

        assert isa(temp, TempType[Attribute])
        temp_lb = None
        temp_ub = None
        if isinstance(temp.bounds, StencilBoundsAttr):
            temp_lb = temp.bounds.lb
            temp_ub = temp.bounds.ub

        temp_lb = IndexAttr.min(op.lb, temp_lb)
        temp_ub = IndexAttr.max(op.ub, temp_ub)
        op.temp.typ = TempType(tuple(zip(temp_lb, temp_ub)), temp.element_type)


class ApplyOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        if len(op.res) < 1:
            return
        res_typ = op.res[0].typ
        assert isa(res_typ, TempType[Attribute])
        assert isinstance(res_typ.bounds, StencilBoundsAttr)
        ntyp = res_typ
        assert isinstance(ntyp.bounds, StencilBoundsAttr)

        accesses = [a for a in op.walk() if isinstance(a, AccessOp)]
        if not accesses:
            return
        for access in accesses:
            temp = access.temp.typ
            assert isa(temp, TempType[Attribute])

            lb = IndexAttr.min(res_typ.bounds.lb + access.offset, ntyp.bounds.lb)
            ub = IndexAttr.max(res_typ.bounds.ub + access.offset, ntyp.bounds.ub)
            ntyp = TempType(tuple(zip(lb, ub)), temp.element_type)
            assert isinstance(ntyp.bounds, StencilBoundsAttr)

        for i, arg in enumerate(op.args):
            if not isa(arg.typ, TempType[Attribute]):
                continue
            arg.typ = ntyp
            op.region.block.args[i].typ = ntyp


ShapeInference = GreedyRewritePatternApplier(
    [
        ApplyOpShapeInference(),
        LoadOpShapeInference(),
        StoreOpShapeInference(),
    ]
)


class StencilShapeInferencePass(ModulePass):
    name = "stencil-shape-inference"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        inference_walker = PatternRewriteWalker(
            ShapeInference, apply_recursively=False, walk_reverse=True
        )
        inference_walker.rewrite_module(op)
