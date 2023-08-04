"""
Temporary workaround to lack of float support in our pipeline.

The rewrites in this file convert arith operations on floats to operations on ints and add
casts between the two.
"""

from typing import cast

from xdsl.dialects import arith
from xdsl.dialects.builtin import (
    AnyFloat,
    FloatAttr,
    IntegerAttr,
    ModuleOp,
    UnrealizedConversionCastOp,
    i32,
)
from xdsl.ir.core import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class CastConstantOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Constant, rewriter: PatternRewriter):
        if isinstance(op.value, IntegerAttr):
            return

        assert isinstance(op.value, FloatAttr)

        op_value = cast(FloatAttr[AnyFloat], op.value)

        value = op.value.value.data
        assert (
            value.is_integer()
        ), f"Only support integer values in arith.Constant, got {value}"
        value = int(value)

        rewriter.replace_matched_op(
            [
                constant := arith.Constant.from_int_and_width(value, 32),
                UnrealizedConversionCastOp.get(constant.results, (op_value.type,)),
            ]
        )


class CastAddfOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addf, rewriter: PatternRewriter):
        rewriter.replace_matched_op(
            [
                lhs := UnrealizedConversionCastOp.get((op.lhs,), (i32,)),
                rhs := UnrealizedConversionCastOp.get((op.rhs,), (i32,)),
                add := arith.Addi(lhs, rhs, i32),
                UnrealizedConversionCastOp.get((add,), (op.result.type,)),
            ]
        )


class CastMulfOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Mulf, rewriter: PatternRewriter):
        rewriter.replace_matched_op(
            [
                lhs := UnrealizedConversionCastOp.get((op.lhs,), (i32,)),
                rhs := UnrealizedConversionCastOp.get((op.rhs,), (i32,)),
                add := arith.Muli(lhs, rhs, i32),
                UnrealizedConversionCastOp.get((add,), (op.result.type,)),
            ]
        )


class CastArithFloatToInt(ModulePass):
    name = "cast-float-to-int"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    CastConstantOp(),
                    CastAddfOp(),
                    CastMulfOp(),
                ]
            )
        ).rewrite_module(op)
