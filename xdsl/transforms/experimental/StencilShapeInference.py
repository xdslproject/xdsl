from typing import Iterable, TypeVar
from xdsl.dialects import builtin
from xdsl.dialects.experimental.stencil import (
    AccessOp,
    ApplyOp,
    HaloSwapOp,
    IndexAttr,
    LoadOp,
    StoreOp,
    TempType,
)
from xdsl.dialects.stencil import CastOp
from xdsl.ir import Attribute, BlockArgument, MLContext, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.experimental.ConvertStencilToLLMLIR import verify_load_bounds
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
        assert apply.lb is not None and apply.ub is not None
        shape_lb = IndexAttr.min(apply.lb, shape_lb)
        shape_ub = IndexAttr.max(apply.ub, shape_ub)

    assert shape_lb is not None and shape_ub is not None
    return shape_lb, shape_ub


class LoadOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        cast = op.field.owner
        assert isinstance(cast, CastOp)

        verify_load_bounds(cast, op)

        assert op.lb and op.ub
        assert isa(op.res.typ, TempType[Attribute])

        # TODO: We need to think about that. Do we want an API for this?
        # Do we just want to recreate the whole operation?
        op.res.typ = TempType(
            IndexAttr.size_from_bounds(op.lb, op.ub),
            op.res.typ.element_type,
        )


class StoreOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter, /):
        owner = op.temp.owner

        assert isinstance(owner, ApplyOp | LoadOp)

        owner.attributes["lb"] = IndexAttr.min(op.lb, owner.lb)
        owner.attributes["ub"] = IndexAttr.max(op.ub, owner.ub)


class ApplyOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        for access in op.walk():
            assert (op.lb is not None) and (op.ub is not None)
            if not isinstance(access, AccessOp):
                continue
            assert isinstance(access.temp, BlockArgument)
            temp_owner = op.args[access.temp.index].owner

            assert isinstance(temp_owner, LoadOp | ApplyOp)

            temp_owner.attributes["lb"] = IndexAttr.min(
                op.lb + access.offset, temp_owner.lb
            )
            temp_owner.attributes["ub"] = IndexAttr.max(
                op.ub + access.offset, temp_owner.ub
            )

        assert op.lb and op.ub

        for result in op.results:
            assert isa(result.typ, TempType[Attribute])
            result.typ = TempType(
                IndexAttr.size_from_bounds(op.lb, op.ub), result.typ.element_type
            )


class HaloOpShapeInference(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: HaloSwapOp, rewriter: PatternRewriter, /):
        assert isinstance(op.input_stencil.owner, LoadOp)
        load = op.input_stencil.owner
        halo_lb, halo_ub = infer_core_size(load)
        op.attributes["core_lb"] = halo_lb
        op.attributes["core_ub"] = halo_ub
        assert load.lb is not None
        assert load.ub is not None
        op.attributes["buff_lb"] = load.lb
        op.attributes["buff_ub"] = load.ub


ShapeInference = GreedyRewritePatternApplier(
    [
        ApplyOpShapeInference(),
        LoadOpShapeInference(),
        StoreOpShapeInference(),
        HaloOpShapeInference(),
    ]
)


class StencilShapeInferencePass(ModulePass):
    name = "stencil-shape-inference"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        inference_walker = PatternRewriteWalker(
            ShapeInference, apply_recursively=False, walk_reverse=True
        )
        inference_walker.rewrite_module(op)
