from xdsl.dialects import memref
from xdsl.ir import Attribute
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


class MemrefSubviewOfSubviewFolding(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Subview, rewriter: PatternRewriter, /):
        source_subview = op.source.owner
        if not isinstance(source_subview, memref.Subview):
            return

        if not all(stride.data == 1 for stride in op.static_strides.data):
            return

        if not all(stride.data == 1 for stride in source_subview.static_strides.data):
            return

        if not len(op.static_offsets.data) == len(source_subview.static_offsets.data):
            return

        assert isa(source_subview.source.type, memref.MemRefType[Attribute])

        assert isa(op.result.type, memref.MemRefType[Attribute])

        reduce_rank = False

        if len(source_subview.source.type.shape) != len(op.result.type.shape):
            reduce_rank = True

        if len(op.offsets) > 0 or len(source_subview.offsets) > 0:
            return
        if len(op.sizes) > 0 or len(source_subview.sizes) > 0:
            return
        if len(op.strides) > 0 or len(source_subview.strides) > 0:
            return

        new_offsets = [
            off1.data + off2.data
            for off1, off2 in zip(
                op.static_offsets.data, source_subview.static_offsets.data
            )
        ]

        current_sizes = [x.data for x in op.static_sizes.data]
        current_strides = [x.data for x in op.static_strides.data]

        assert isa(new_offsets, list[int])
        assert isa(current_sizes, list[int])
        assert isa(current_strides, list[int])

        new_op = memref.Subview.from_static_parameters(
            source_subview.source,
            source_subview.source.type,
            new_offsets,
            current_sizes,
            current_strides,
            reduce_rank=reduce_rank,
        )
        if reduce_rank:
            if new_op.result.type != op.result.type:
                return

        rewriter.replace_matched_op(new_op)


class ElideUnusedAlloc(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Alloc, rewriter: PatternRewriter, /):
        if len(op.memref.uses) == 1 and isinstance(
            only_use := tuple(op.memref.uses)[0].operation, memref.Dealloc
        ):
            rewriter.erase_op(only_use)
            rewriter.erase_matched_op()
