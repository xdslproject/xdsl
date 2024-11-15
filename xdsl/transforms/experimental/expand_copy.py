# Port of the expand-copy pass in Google's HEIR: https://heir.dev/docs/passes/#-expand-copy

from dataclasses import dataclass

from xdsl.builder import Builder, InsertPoint
from xdsl.context import MLContext
from xdsl.dialects import affine, builtin, memref
from xdsl.dialects.affine import AffineExpr, AffineMap, AffineMapAttr
from xdsl.ir import BlockArgument
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class ExpandCopy(RewritePattern):
    ########################################################################################
    # Expands memref.copy to a nested structure of affine loops with load/store.
    ########################################################################################

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.CopyOp, rewriter: PatternRewriter):
        in_buf = op.source
        out_buf = op.destination

        assert isinstance(in_buf.type, builtin.MemRefType)
        ndims = len(in_buf.type.shape)
        shape = in_buf.type.shape
        res_type = in_buf.type.element_type

        # Generate the loop nesting structure top-bottom down to the innermost loop
        loopnest_queue: list[affine.For] = []

        for idx in range(ndims - 1):

            @Builder.region([builtin.IndexType()])
            def loop_body(builder: Builder, args: [BlockArgument, ...]):
                builder.insert(affine.Yield.get())

            for_loop = affine.For.from_region(
                [], [], [], [], 0, shape.data[idx].data, loop_body
            )
            if loopnest_queue:
                rewriter.insert_op(
                    for_loop, InsertPoint.at_start(loopnest_queue[-1].body.block)
                )

            loopnest_queue.append(for_loop)

        # Generate the innermost loop that performs element-wise copy and insert in the nesting structure
        @Builder.region([builtin.IndexType()])
        def innermost_loop_body(builder: Builder, args: [BlockArgument, ...]):
            for_indices = list(
                map(lambda loop: loop.body.block.args[0], loopnest_queue)
            )
            for_indices.append(args[0])

            dims = [AffineExpr.dimension(dim) for dim in range(ndims)]
            aff_map = AffineMapAttr(AffineMap(ndims, 0, dims))
            load = affine.Load(in_buf, for_indices, aff_map, res_type)
            store = affine.Store(load.result, out_buf, for_indices, aff_map)

            builder.insert(load)
            builder.insert(store)
            builder.insert(affine.Yield.get())

        innermost_loop = affine.For.from_region(
            [], [], [], [], 0, shape.data[-1].data, innermost_loop_body
        )
        rewriter.insert_op(
            innermost_loop, InsertPoint.at_start(loopnest_queue[-1].body.block)
        )

        top_loop = loopnest_queue[0]
        rewriter.replace_matched_op(top_loop)


@dataclass(frozen=True)
class ExpandCopyPass(ModulePass):
    name = "expand-copy"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        expand_copy_pass = PatternRewriteWalker(
            ExpandCopy(),
            apply_recursively=False,
            walk_reverse=False,
        )
        expand_copy_pass.rewrite_module(op)
