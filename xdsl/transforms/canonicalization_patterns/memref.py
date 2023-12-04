import sys

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
            print(
                "bail: not isinstance(source_subview, memref.Subview)", file=sys.stderr
            )
            return

        if not all(stride.data == 1 for stride in op.static_strides.data):
            print(
                "bail: not all(stride.data == 1 for stride in op.static_strides.data)",
                file=sys.stderr,
            )
            return

        if not all(stride.data == 1 for stride in source_subview.static_strides.data):
            print(
                "bail: not all(stride.data == 1 for stride in source_subview.static_strides.data)",
                file=sys.stderr,
            )
            return

        if not len(op.static_offsets.data) == len(source_subview.static_offsets.data):
            print(
                "bail: not len(op.static_offsets.data) == len(source_subview.static_offsets.data)",
                file=sys.stderr,
            )
            return

        assert isa(source_subview.source.type, memref.MemRefType[Attribute])

        assert isa(op.result.type, memref.MemRefType[Attribute])

        reduce_rank = False

        if len(source_subview.source.type.shape) != len(op.result.type.shape):
            print(
                "bail: len(source_subview.source.type.shape) != len(op.result.type.shape)",
                file=sys.stderr,
            )
            print("lets not bail outright", file=sys.stderr)
            reduce_rank = True

        if len(op.offsets) > 0 or len(source_subview.offsets) > 0:
            print(
                "bail: len(op.offsets) > 0 or len(source_subview.offsets) > 0",
                file=sys.stderr,
            )
            return
        if len(op.sizes) > 0 or len(source_subview.sizes) > 0:
            print(
                "bail: len(op.sizes) > 0 or len(source_subview.sizes) > 0",
                file=sys.stderr,
            )
            return
        if len(op.strides) > 0 or len(source_subview.strides) > 0:
            print(
                "bail: len(op.strides) > 0 or len(source_subview.strides) > 0",
                file=sys.stderr,
            )
            return

        new_offsets = [
            off1.data + off2.data
            for off1, off2 in zip(
                op.static_offsets.data, source_subview.static_offsets.data
            )
        ]

        new_op = memref.Subview.from_static_parameters(
            source_subview.source,
            source_subview.source.type,
            new_offsets,
            tuple(x.data for x in op.static_sizes.data),
            tuple(x.data for x in op.static_strides.data),
            reduce_rank=reduce_rank,
        )
        if reduce_rank:
            if new_op.result.type != op.result.type:
                print("bail: new_op.result.type != op.result.type", file=sys.stderr)
                return

        rewriter.replace_matched_op(new_op)
