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


@dataclass(frozen=True)
class FuncOpShapeUpdate(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        op.update_function_type()


@dataclass(frozen=True)
class StencilShapeMinimize(ModulePass):
    """
    Minimises the shapes of `stencil.field` types that have been over-allocated and are larger than necessary.
    """

    name = "stencil-shape-minimize"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
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
