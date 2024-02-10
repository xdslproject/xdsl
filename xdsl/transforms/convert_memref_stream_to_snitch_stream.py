from typing import cast

from xdsl.backend.riscv.lowering.utils import (
    register_type_for_type,
)
from xdsl.dialects import memref_stream, riscv_snitch, stream
from xdsl.dialects.builtin import (
    ModuleOp,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Attribute, MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class ReadOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.ReadOp, rewriter: PatternRewriter
    ) -> None:
        stream_type = op.stream.type
        assert isinstance(stream_type, stream.ReadableStreamType)
        value_type = cast(
            stream.ReadableStreamType[Attribute], stream_type
        ).element_type
        register_type = register_type_for_type(value_type).unallocated()

        rewriter.replace_matched_op(
            (
                new_stream := UnrealizedConversionCastOp.get(
                    (op.stream,), (stream.ReadableStreamType(register_type),)
                ),
                new_op := riscv_snitch.ReadOp(new_stream.results[0]),
                UnrealizedConversionCastOp.get(
                    (new_op.res,),
                    (value_type,),
                ),
            ),
        )


class WriteOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.WriteOp, rewriter: PatternRewriter
    ) -> None:
        stream_type = op.stream.type
        assert isinstance(stream_type, stream.WritableStreamType)
        value_type = cast(
            stream.WritableStreamType[Attribute], stream_type
        ).element_type
        register_type = register_type_for_type(value_type).unallocated()

        rewriter.replace_matched_op(
            (
                new_stream := UnrealizedConversionCastOp.get(
                    (op.stream,), (stream.WritableStreamType(register_type),)
                ),
                new_value := UnrealizedConversionCastOp.get(
                    (op.value,), (register_type,)
                ),
                riscv_snitch.WriteOp(new_value.results[0], new_stream.results[0]),
            ),
        )


class ConvertMemrefStreamToSnitch(ModulePass):
    name = "convert-memref-stream-to-snitch"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ReadOpLowering(),
                    WriteOpLowering(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
