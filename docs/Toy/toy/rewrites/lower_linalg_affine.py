"""
This file implements a partial lowering of Toy operations to a combination of
affine loops, memref operations and standard operations. This lowering
expects that all calls have been inlined, and all shapes have been resolved.
"""

from xdsl.builder import Builder
from xdsl.dialects import affine, linalg
from xdsl.dialects.builtin import (
    ModuleOp,
)
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from .lower_toy_affine import (
    ValueRange,
    build_affine_loop_nest_const,
)

# region Helpers


class GenericLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        # Create a nest of affine loops, with one loop per dimension of the shape.
        # The buildAffineLoopNest function takes a callback that is used to construct the body
        # of the innermost loop given a builder, a location and a range of loop induction
        # variables.

        rank = op.get_num_loops()
        lower_bounds = tuple(0 for _ in range(rank))
        steps = tuple(1 for _ in range(rank))

        def impl_loop(nested_builder: Builder, ivs: ValueRange):
            load_ops = [
                nested_builder.insert(
                    affine.Load(op.inputs[i], ivs, op.indexing_maps.data[i])
                )
                for i in range(len(op.inputs))
            ]

            block = op.body.block

            # Replace block args with operand casts
            for load_op, arg in zip(load_ops, block.args):
                arg.replace_by(load_op.result)

            # remove block args
            while len(block.args):
                assert not len(block.args[-1].uses)
                rewriter.erase_block_argument(block.args[-1])

            # Inline yield body after loads
            rewriter.inline_block_after(block, load_ops[-1])

            # Get return from function definition
            body_block = load_ops[-1].parent
            assert body_block is not None
            yield_op = body_block.last_op
            assert isinstance(yield_op, linalg.Yield), f"{yield_op}"

            store_op = affine.Store(
                yield_op.values[0], op.outputs[0], ivs, op.indexing_maps.data[-1]
            )
            nested_builder.insert(store_op)
            rewriter.erase_op(yield_op)

        parent_block = op.parent
        assert parent_block is not None
        builder = Builder(parent_block, op)
        build_affine_loop_nest_const(
            builder, lower_bounds, op.get_static_loop_ranges(), steps, impl_loop
        )

        rewriter.erase_matched_op()


# endregion RewritePatterns


class LinalgToAffinePass(ModulePass):
    name = "linalg-to-affine"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    GenericLowering(),
                ]
            )
        ).rewrite_module(op)
