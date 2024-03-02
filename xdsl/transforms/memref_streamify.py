from dataclasses import dataclass, field
from typing import cast

from xdsl.dialects import memref, memref_stream, stream
from xdsl.dialects.builtin import (
    ArrayAttr,
    ModuleOp,
)
from xdsl.ir import Attribute, Block, MLContext, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class StreamifyGenericOpPattern(RewritePattern):
    streams: int = field()

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.GenericOp, rewriter: PatternRewriter
    ) -> None:
        # Currently can only stream memrefs that are not inout
        streamable_input_indices = tuple(
            (index, cast(memref.MemRefType[Attribute], value_type).element_type)
            for index, value in enumerate(op.inputs)
            if isinstance(value_type := value.type, memref.MemRefType)
        )
        input_count = len(op.inputs)
        streamable_output_indices = tuple(
            (index, cast(memref.MemRefType[Attribute], value_type).element_type)
            for index, value in enumerate(op.outputs)
            if isinstance(value.type, memref.MemRefType)
            if not op.body.block.args[index + input_count].uses
        )
        # We might want to pick which memref to stream by iteration count in the future
        streamed_input_indices = streamable_input_indices[: self.streams]
        streamed_output_indices = streamable_output_indices[
            : self.streams - len(streamed_input_indices)
        ]
        streamed_operand_indices = streamed_input_indices + tuple(
            (index + len(streamed_input_indices), el_type)
            for index, el_type in streamed_output_indices
        )
        input_el_types = tuple(el_type for _, el_type in streamed_input_indices)
        output_el_types = tuple(el_type for _, el_type in streamed_output_indices)
        input_stream_types = tuple(
            stream.ReadableStreamType(el_type) for el_type in input_el_types
        )
        output_stream_types = tuple(
            stream.WritableStreamType(el_type) for el_type in output_el_types
        )
        rewriter.insert_op_before_matched_op(
            streaming_region_op := memref_stream.StreamingRegionOp(
                tuple(op.inputs[index] for index, _ in streamed_input_indices),
                tuple(op.outputs[index] for index, _ in streamable_output_indices),
                ArrayAttr(
                    tuple(
                        op.indexing_maps.data[index]
                        for index, _ in streamed_operand_indices
                    )
                ),
                op.bounds,
                Region(Block(arg_types=input_stream_types + output_stream_types)),
            )
        )
        new_body = streaming_region_op.body.block
        new_operands = list(op.operands)
        for stream_index, (index, _) in enumerate(streamed_operand_indices):
            new_operands[index] = new_body.args[stream_index]

        rewriter.insert_op_at_end(
            memref_stream.GenericOp(
                new_operands[:input_count],
                new_operands[input_count:],
                rewriter.move_region_contents_to_new_regions(op.body),
                op.indexing_maps,
                op.iterator_types,
                op.bounds,
            ),
            new_body,
        )
        rewriter.erase_matched_op()


@dataclass(frozen=True)
class MemrefStreamifyPass(ModulePass):
    """
    Converts a memref generic on memrefs to a memref generic on streams, by moving it into
    a streaming region.
    """

    name = "memref-streamify"

    streams: int = field(default=3)

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            StreamifyGenericOpPattern(self.streams),
            apply_recursively=False,
        ).rewrite_module(op)
