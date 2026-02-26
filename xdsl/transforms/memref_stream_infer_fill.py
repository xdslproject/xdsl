from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import memref, memref_stream
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class InferFillPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.GenericOp, rewriter: PatternRewriter
    ) -> None:
        if len(op.inputs) != 1:
            return

        if len(op.outputs) != 1:
            return

        if op.inits:
            return

        if any(
            iterator_type.data != memref_stream.IteratorType.PARALLEL
            for iterator_type in op.iterator_types.data
        ):
            return

        output = op.outputs[0]
        input = op.inputs[0]

        if not isinstance(output_type := output.type, memref.MemRefType):
            return

        output_type = cast(memref.MemRefType, output.type)

        type_shape = output_type.get_shape()
        bounds = tuple(attr.value.data for attr in op.bounds)

        if type_shape != bounds:
            return

        if input.type != output_type.element_type:
            return

        block = op.body.block
        ops = tuple(block.ops)

        if len(ops) != 1:
            return

        if not isinstance(yield_op := ops[0], memref_stream.YieldOp):
            return

        if len(yielded_vals := tuple(yield_op.operands)) != 1:
            return

        if yielded_vals[0] is not block.args[0]:
            return

        rewriter.replace_op(op, memref_stream.FillOp(output, input))


@dataclass(frozen=True)
class MemRefStreamInferFillPass(ModulePass):
    """
    Detects memref_stream.generic operations that can be represented as
    `memref_stream.fill` ops.
    """

    name = "memref-stream-infer-fill"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            InferFillPattern(),
            apply_recursively=False,
        ).rewrite_module(op)
