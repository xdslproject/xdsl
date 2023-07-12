from typing import Iterable, TypeVar, cast

from xdsl.dialects import builtin
from xdsl.dialects.stencil import (
    AccessOp,
    ApplyOp,
    BufferOp,
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

    assert shape_lb is not None and shape_ub is not None
    return shape_lb, shape_ub


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
        ApplyOpShapeInference(),
        BufferOpShapeInference(),
        AccessOpShapeInference(),
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
