from typing import Any, cast

from xdsl.dialects import bufferization, memref, ml_program
from xdsl.dialects.builtin import (
    ModuleOp,
    TensorType,
    UnitAttr,
)
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class ConvertGlobalPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: ml_program.Global, rewriter: PatternRewriter
    ) -> None:
        if op.value is None:
            raise NotImplementedError(
                "Converting ml_program.global with no value not implemented"
            )
        assert isinstance(op_type := op.type, TensorType)
        op_type = cast(TensorType[Any], op_type)
        new_type = memref.MemRefType(op_type.element_type, op_type.shape)
        rewriter.replace_matched_op(
            (
                memref.Global.get(
                    op.sym_name,
                    new_type,
                    op.value,
                    op.sym_visibility,
                    UnitAttr() if op.is_mutable is None else None,
                ),
            )
        )


class ConvertGlobalLoadConst(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: ml_program.GlobalLoadConstant, rewriter: PatternRewriter
    ) -> None:
        assert isinstance(op_type := op.result.type, TensorType)
        op_type = cast(TensorType[Any], op_type)
        new_type = memref.MemRefType(op_type.element_type, op_type.shape)
        rewriter.replace_matched_op(
            (
                mem := memref.GetGlobal.get(op.global_attr, new_type),
                bufferization.ToTensorOp(mem.memref),
            )
        )


class ConvertMlProgramToMemrefPass(ModulePass):
    """
    Converts operations in the `ml_program` dialect to `memref`.
    `ml_program` operations are at the `tensor` level of abstraction, so some of the
    rewrites insert `bufferization` ops to bridge the gap to existing consumers of global
    `tensor`s.
    """

    name = "convert-ml-program-to-memref"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertGlobalPattern(),
                    ConvertGlobalLoadConst(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
