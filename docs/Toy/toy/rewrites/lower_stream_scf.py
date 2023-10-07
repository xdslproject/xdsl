from xdsl.builder import Builder
from xdsl.dialects import stream
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


class LowerGenericOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stream.GenericOp, rewriter: PatternRewriter):
        # Create a nest of affine loops, with one loop per dimension of the shape.
        # The buildAffineLoopNest function takes a callback that is used to construct the body
        # of the innermost loop given a builder, a location and a range of loop induction
        # variables.

        ub = tuple(b.data for b in op.static_loop_ranges.data)
        lower_bounds = tuple(0 for _ in range(len(ub)))
        steps = tuple(1 for _ in range(len(ub)))

        def impl_loop(nested_builder: Builder, ivs: ValueRange):
            load_ops = [nested_builder.insert(stream.ReadOp(i)) for i in op.inputs]

            block = op.body.block

            # Replace block args with operand casts
            for load_op, arg in zip(load_ops, block.args):
                arg.replace_by(load_op.res)

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
            assert isinstance(yield_op, stream.YieldOp), f"{yield_op}"

            for v, o in zip(yield_op.operands, op.outputs):
                nested_builder.insert(stream.WriteOp(v, o))

            rewriter.erase_op(yield_op)

        parent_block = op.parent
        assert parent_block is not None
        builder = Builder(parent_block, op)
        build_affine_loop_nest_const(builder, lower_bounds, ub, steps, impl_loop)

        rewriter.erase_matched_op()


class StreamToScfPass(ModulePass):
    name = "stream-to-scf"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerGenericOp(),
                ]
            )
        ).rewrite_module(op)
