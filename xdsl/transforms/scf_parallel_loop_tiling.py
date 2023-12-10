from dataclasses import dataclass

from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntegerAttr, ModuleOp
from xdsl.dialects.scf import ParallelOp, Yield
from xdsl.ir import Block, MLContext, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


@dataclass
class ScfParallelLoopTilingPattern(RewritePattern):
    tile_sizes: list[int]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ParallelOp, rewriter: PatternRewriter, /):
        lower = tuple(o.owner for o in op.lowerBound)
        upper = tuple(o.owner for o in op.upperBound)
        step = tuple(o.owner for o in op.step)

        # Only handle constant loop bounds for now
        if not (
            isa(lower, tuple[arith.Constant, ...])
            and isa(upper, tuple[arith.Constant, ...])
            and isa(step, tuple[arith.Constant, ...])
        ):
            print("not constant")
            print(lower)
            print(upper)
            print(step)
            return
        lower_v = tuple(o.value for o in lower)
        upper_v = tuple(o.value for o in upper)
        step_v = tuple(o.value for o in step)

        # Verifier ensures those are of index type
        assert isa(lower_v, tuple[IntegerAttr[IndexType], ...])
        assert isa(upper_v, tuple[IntegerAttr[IndexType], ...])
        assert isa(step_v, tuple[IntegerAttr[IndexType], ...])

        # Get the values
        lower_v = tuple(o.value.data for o in lower_v)
        upper_v = tuple(o.value.data for o in upper_v)
        step_v = tuple(o.value.data for o in step_v)

        # fill the tile sizes with ones
        tile_sizes = self.tile_sizes
        for _ in range(len(tile_sizes), len(lower)):
            tile_sizes.append(1)

        # Only handle perfectly divisible loops for now
        if any((u - l) % s != 0 for l, u, s in zip(lower_v, upper_v, tile_sizes)):
            print("not divisible")
            return

        zero = arith.Constant(IntegerAttr.from_index_int_value(0))
        tile_sizes = {i: s for i, s in enumerate(tile_sizes) if s != 0}
        tile_sizes = {
            i: arith.Constant(IntegerAttr.from_index_int_value(s))
            for i, s in tile_sizes.items()
        }
        tiled_dims = sorted(tile_sizes.keys())
        outter_lower = [lower[d] for d in tiled_dims]
        outter_upper = [upper[d] for d in tiled_dims]
        outter_step = [arith.Muli(step[d], tile_sizes[d]) for d in tiled_dims]

        inner_lower = [zero] * len(op.lowerBound)
        inner_upper = [
            tile_sizes[i] if i in tile_sizes else upper[i]
            for i in range(len(op.lowerBound))
        ]
        inner_loop = ParallelOp(inner_lower, inner_upper, step, op.detach_region(0))
        outter_loop = ParallelOp(
            outter_lower,
            outter_upper,
            outter_step,
            Region(
                Block(
                    [inner_loop, Yield()], arg_types=[IndexType()] * len(outter_lower)
                )
            ),
        )
        print("tiled")
        rewriter.replace_matched_op(
            [zero, *tile_sizes.values(), *outter_step, outter_loop]
        )


@dataclass
class ScfParallelLoopTilingPass(ModulePass):
    name = "scf-parallel-loop-tiling"

    parallel_loop_tile_sizes: list[int]

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [ScfParallelLoopTilingPattern(self.parallel_loop_tile_sizes)]
            ),
            walk_regions_first=True,
            apply_recursively=False,
        )
        walker.rewrite_module(op)
