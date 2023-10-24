from xdsl.dialects import arith, scf, stream
from xdsl.dialects.builtin import (
    IndexType,
    IntegerAttr,
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


class LowerGenericOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stream.GenericOp, rewriter: PatternRewriter):
        # Create a nest of affine loops, with one loop per dimension of the shape.
        # The buildAffineLoopNest function takes a callback that is used to construct the body
        # of the innermost loop given a builder, a location and a range of loop induction
        # variables.

        index = IndexType()

        rewriter.replace_matched_op(
            [
                lb := arith.Constant(IntegerAttr(0, index)),
                step := arith.Constant(IntegerAttr(1, index)),
                loop := scf.For(
                    lb,
                    op.repeat_count,
                    step,
                    (),
                    rewriter.move_region_contents_to_new_regions(op.body),
                ),
            ]
        )

        load_ops = [stream.ReadOp(i) for i in op.inputs]

        block = loop.body.block

        # Add Read ops to block
        for load_op in reversed(load_ops):
            rewriter.insert_op_at_start(load_op, block)

        # Replace block args with operand casts
        for load_op, arg in zip(load_ops, block.args):
            arg.replace_by(load_op.res)

        # remove block args
        while len(block.args):
            assert not len(block.args[-1].uses)
            rewriter.erase_block_argument(block.args[-1])

        # Get return from function definition
        body_block = load_ops[-1].parent
        assert body_block is not None
        yield_op = body_block.last_op
        assert isinstance(yield_op, stream.YieldOp), f"{yield_op}"

        for v, o in zip(yield_op.operands, op.outputs):
            rewriter.insert_op_before(stream.WriteOp(v, o), yield_op)

        rewriter.erase_op(yield_op)


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
