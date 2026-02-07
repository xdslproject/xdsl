from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects import memref, memref_stream
from xdsl.dialects.builtin import ArrayAttr, ModuleOp
from xdsl.ir import Block, Region
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
        if any(
            isinstance(
                operand.type,
                memref_stream.ReadableStreamType | memref_stream.WritableStreamType,
            )
            for operand in op.operands
        ):
            # Already streamified
            return

        init_indices = set(index.data for index in op.init_indices)

        # Can only stream memrefs that are not inout
        input_count = len(op.inputs)
        streamable_input_indices = tuple(
            (index, arg.type)
            for index, (i, arg) in enumerate(
                zip(op.inputs, op.body.block.args[:input_count])
            )
            if isinstance(i_type := i.type, memref.MemRefType) and arg.uses
            if i_type.get_shape()
        )
        streamable_output_indices = tuple(
            (index, arg.type)
            for index, (o, arg) in enumerate(
                zip(op.outputs, op.body.block.args[input_count:])
            )
            if isinstance(o_type := o.type, memref.MemRefType)
            if index in init_indices or not arg.uses
            if o_type.get_shape()
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
            memref_stream.ReadableStreamType(el_type) for el_type in input_el_types
        )
        output_stream_types = tuple(
            memref_stream.WritableStreamType(el_type) for el_type in output_el_types
        )

        # input patterns are never unnested
        input_patterns = tuple(
            memref_stream.StridePattern(
                op.bounds,
                indexing_map,
            )
            for index, _ in streamable_input_indices
            if (indexing_map := op.indexing_maps.data[index])
        )
        # output patterns never contain iteration dimensions
        output_patterns = tuple(
            memref_stream.StridePattern(
                ArrayAttr(
                    tuple(
                        bound
                        for iterator_type, bound in zip(
                            op.iterator_types, op.bounds.data
                        )
                        if iterator_type.data != memref_stream.IteratorType.REDUCTION
                    )
                ),
                indexing_map,
            )
            for output_index, _ in streamed_output_indices
            if (indexing_map := op.indexing_maps.data[output_index + input_count])
        )

        patterns = ArrayAttr(input_patterns + output_patterns)
        rewriter.insert_op(
            streaming_region_op := memref_stream.StreamingRegionOp(
                tuple(op.inputs[index] for index, _ in streamed_input_indices),
                tuple(op.outputs[index] for index, _ in streamable_output_indices),
                patterns,
                Region(Block(arg_types=input_stream_types + output_stream_types)),
            )
        )
        new_body = streaming_region_op.body.block
        new_operands = list(op.operands[: len(op.inputs) + len(op.outputs)])
        for stream_index, (index, _) in enumerate(streamed_operand_indices):
            new_operands[index] = new_body.args[stream_index]

        rewriter.insert_op(
            memref_stream.GenericOp(
                new_operands[:input_count],
                new_operands[input_count:],
                op.inits,
                rewriter.move_region_contents_to_new_regions(op.body),
                op.indexing_maps,
                op.iterator_types,
                op.bounds,
                op.init_indices,
                op.doc,
                op.library_call,
            ),
            InsertPoint.at_end(new_body),
        )
        rewriter.erase_op(op)


@dataclass(frozen=True)
class MemRefStreamifyPass(ModulePass):
    """
    Converts a memref generic on memrefs to a memref generic on streams, by moving it
    into a streaming region.
    """

    name = "memref-streamify"

    streams: int = field(default=3)

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            StreamifyGenericOpPattern(self.streams),
            apply_recursively=False,
        ).rewrite_module(op)
