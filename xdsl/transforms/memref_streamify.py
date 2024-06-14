from dataclasses import dataclass, field
from typing import cast

from xdsl.context import MLContext
from xdsl.dialects import memref, memref_stream, stream
from xdsl.dialects.builtin import ArrayAttr, ModuleOp, UnitAttr
from xdsl.ir import Attribute, Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


@dataclass
class StreamifyGenericOpPattern(RewritePattern):
    streams: int = field()

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.GenericOp, rewriter: PatternRewriter
    ) -> None:
        if any(isinstance(operand.type, stream.StreamType) for operand in op.operands):
            # Already streamified
            return

        if any(not isinstance(init, UnitAttr) for init in op.inits):
            raise NotImplementedError(
                "Cannot streamify operation that has inits that are not UnitAttr"
            )

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
            if isinstance(value_type := value.type, memref.MemRefType)
            if not op.body.block.args[index + input_count].uses
        )
        if not streamable_input_indices and not streamable_output_indices:
            # No memrefs to convert to streams
            return
        # We might want to pick which memref to stream by iteration count in the future
        streamed_input_indices = streamable_input_indices[: self.streams]
        streamed_output_indices = streamable_output_indices[
            : self.streams - len(streamed_input_indices)
        ]
        streamed_operand_indices = streamed_input_indices + tuple(
            (index + input_count, el_type) for index, el_type in streamed_output_indices
        )
        input_el_types = tuple(el_type for _, el_type in streamed_input_indices)
        output_el_types = tuple(el_type for _, el_type in streamed_output_indices)
        input_stream_types = tuple(
            stream.ReadableStreamType(el_type) for el_type in input_el_types
        )
        output_stream_types = tuple(
            stream.WritableStreamType(el_type) for el_type in output_el_types
        )

        patterns = ArrayAttr(
            tuple(
                memref_stream.StridePattern(
                    ArrayAttr(op.bounds.data[: indexing_map.data.num_dims]),
                    indexing_map,
                )
                for index, _ in streamed_operand_indices
                if (indexing_map := op.indexing_maps.data[index])
            )
        )
        rewriter.insert_op_before_matched_op(
            streaming_region_op := memref_stream.StreamingRegionOp(
                tuple(op.inputs[index] for index, _ in streamed_input_indices),
                tuple(op.outputs[index] for index, _ in streamable_output_indices),
                patterns,
                Region(Block(arg_types=input_stream_types + output_stream_types)),
            )
        )
        new_body = streaming_region_op.body.block
        new_operands = list(op.operands)
        for stream_index, (index, _) in enumerate(streamed_operand_indices):
            new_operands[index] = new_body.args[stream_index]

        rewriter.insert_op(
            memref_stream.GenericOp(
                new_operands[:input_count],
                new_operands[input_count:],
                rewriter.move_region_contents_to_new_regions(op.body),
                op.inits,
                op.indexing_maps,
                op.iterator_types,
                op.bounds,
            ),
            InsertPoint.at_end(new_body),
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
