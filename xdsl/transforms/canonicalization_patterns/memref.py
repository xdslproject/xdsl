from xdsl.dialects import memref
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


class MemRefSubviewOfSubviewFolding(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.SubviewOp, rewriter: PatternRewriter, /):
        source_subview = op.source.owner
        if not isinstance(source_subview, memref.SubviewOp):
            return

        current_strides = op.static_strides.get_values()

        if not all(stride == 1 for stride in current_strides):
            return

        if not all(
            stride == 1 for stride in source_subview.static_strides.iter_values()
        ):
            return

        if not len(op.static_offsets) == len(source_subview.static_offsets):
            return

        assert isa(source_subview.source.type, memref.MemRefType)

        assert isa(op.result.type, memref.MemRefType)

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
            off1 + off2
            for off1, off2 in zip(
                op.static_offsets.iter_values(),
                source_subview.static_offsets.iter_values(),
                strict=True,
            )
        ]

        current_sizes = op.static_sizes.get_values()

        new_op = memref.SubviewOp.from_static_parameters(
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

        rewriter.replace_op(op, new_op)


class ElideUnusedAlloc(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.AllocOp, rewriter: PatternRewriter, /):
        if isinstance(only_use := op.memref.get_user_of_unique_use(), memref.DeallocOp):
            rewriter.erase_op(only_use)
            rewriter.erase_op(op)
