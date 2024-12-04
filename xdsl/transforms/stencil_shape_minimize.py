from dataclasses import dataclass, field

from xdsl.context import MLContext
from xdsl.dialects import builtin, func, stencil
from xdsl.ir import Attribute
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.shape_inference import infer_shapes


@dataclass
class ShapeAnalysis(TypeConversionPattern):
    seen: set[stencil.TempType[Attribute]] = field(default_factory=set)

    @attr_type_rewrite_pattern
    def convert_type(self, typ: stencil.TempType[Attribute], /) -> Attribute | None:
        self.seen.add(typ)


@dataclass
class ShapeMinimisation(TypeConversionPattern):
    shape: stencil.StencilBoundsAttr | None = None

    @attr_type_rewrite_pattern
    def convert_type(self, typ: stencil.FieldType[Attribute], /) -> Attribute | None:
        if typ.bounds != self.shape and self.shape:
            return stencil.FieldType(self.shape, typ.element_type)


@dataclass
class InvalidateTemps(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: stencil.TempType[Attribute], /) -> Attribute | None:
        if isinstance(typ.bounds, stencil.StencilBoundsAttr):
            return stencil.TempType(len(typ.bounds.lb), typ.element_type)


@dataclass(frozen=True)
class FuncOpShapeUpdate(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        if not op.is_declaration:
            op.update_function_type()


@dataclass
class RestrictStoreOp(RewritePattern):
    restrict: tuple[int, ...]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.StoreOp, rewriter: PatternRewriter, /):
        new_bounds = [
            (min(lower_bound, bound_lim), min(upper_bound, bound_lim))
            for lower_bound, upper_bound, bound_lim in zip(
                op.bounds.lb, op.bounds.ub, self.restrict
            )
        ]
        new_bounds_attr = stencil.StencilBoundsAttr(new_bounds)
        if new_bounds_attr != op.bounds:
            rewriter.replace_matched_op(
                stencil.StoreOp.get(
                    temp=op.temp, field=op.field, bounds=new_bounds_attr
                )
            )


@dataclass(frozen=True)
class StencilShapeMinimize(ModulePass):
    """
    Minimises the shapes of `stencil.field` types that have been over-allocated and are larger than necessary.
    """

    name = "stencil-shape-minimize"

    restrict: tuple[int, ...] | None = None

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        if self.restrict:
            PatternRewriteWalker(
                GreedyRewritePatternApplier(
                    [
                        InvalidateTemps(),
                        RestrictStoreOp(restrict=self.restrict),
                    ]
                )
            ).rewrite_module(op)
            infer_shapes(op)
        analysis = ShapeAnalysis(seen=set())
        PatternRewriteWalker(analysis).rewrite_module(op)
        bounds = set(
            t.bounds
            for t in analysis.seen
            if isinstance(t.bounds, stencil.StencilBoundsAttr)
        )
        if not bounds:
            return
        shape: stencil.StencilBoundsAttr = bounds.pop()
        for b in bounds:
            shape = shape | b

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ShapeMinimisation(shape=shape),
                    FuncOpShapeUpdate(),
                ]
            )
        ).rewrite_module(op)
