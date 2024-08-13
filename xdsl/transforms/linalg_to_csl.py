from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import linalg
from xdsl.dialects.builtin import AnyMemRefType, Float16Type, Float32Type, ModuleOp
from xdsl.dialects.csl import csl
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


class ConvertBinaryLinalgOp(RewritePattern):
    """
    Base class for converting binary linalg operations.
    """

    def transform_op(
        self,
        op: linalg.NamedOpBase,
        rewriter: PatternRewriter,
        f16: type[csl.BuiltinDsdOp],
        f32: type[csl.BuiltinDsdOp],
    ):
        if not isa(op.outputs.types[0], AnyMemRefType):
            return

        match op.outputs.types[0].get_element_type():
            case Float16Type():
                builtin = f16
            case Float32Type():
                builtin = f32
            case _:
                raise ValueError(
                    f"Unsupported element type {op.outputs.types[0].get_element_type()}"
                )

        rewriter.replace_matched_op(
            builtin(
                operands=[
                    [
                        op.outputs[0],
                        op.inputs[0],
                        op.inputs[1],
                    ]
                ]
            )
        )


class ConvertLinalgAddPass(ConvertBinaryLinalgOp):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.AddOp, rewriter: PatternRewriter, /):
        self.transform_op(op, rewriter, f16=csl.FaddhOp, f32=csl.FaddsOp)


class ConvertLinalgSubPass(ConvertBinaryLinalgOp):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.SubOp, rewriter: PatternRewriter, /):
        self.transform_op(op, rewriter, f16=csl.FsubhOp, f32=csl.FsubsOp)


class ConvertLinalgMulPass(ConvertBinaryLinalgOp):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.MulOp, rewriter: PatternRewriter, /):
        self.transform_op(op, rewriter, f16=csl.FmulhOp, f32=csl.FmulsOp)


@dataclass(frozen=True)
class LinalgToCsl(ModulePass):
    """
    Convert linalg ops to csl ops.

    The linalg ops are required to be in 'memref mode', i.e., after bufferization has been applied.
    """

    name = "linalg-to-csl"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertLinalgAddPass(),
                    ConvertLinalgSubPass(),
                    ConvertLinalgMulPass(),
                ]
            ),
        )
        module_pass.rewrite_module(op)
