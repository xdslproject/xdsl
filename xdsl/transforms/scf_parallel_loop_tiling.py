from dataclasses import dataclass

from xdsl.dialects import affine, arith
from xdsl.dialects.builtin import IndexType, IntegerAttr, ModuleOp
from xdsl.dialects.scf import ParallelOp, Yield
from xdsl.ir import Block, MLContext, Operation, Region
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
        # Only tile the innermost parallel loop
        if any(isinstance(o, ParallelOp) for o in op.body.walk()):
            return
        lower = tuple(o.owner for o in op.lowerBound)
        upper = tuple(o.owner for o in op.upperBound)
        step = tuple(o.owner for o in op.step)

        # Only handle constant loop bounds for now
        if not (
            isa(lower, tuple[arith.Constant, ...])
            and isa(upper, tuple[arith.Constant, ...])
            and isa(step, tuple[arith.Constant, ...])
        ):
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
        tile_sizes_v = self.tile_sizes
        for _ in range(len(tile_sizes_v), len(lower)):
            tile_sizes_v.append(1)

        # Only handle perfectly divisible loops for now
        # if any((u - l) % s != 0 for l, u, s in zip(lower_v, upper_v, tile_sizes_v)):
        #     return

        zero = arith.Constant(IntegerAttr.from_index_int_value(0))
        tile_sizes_v = {i: s for i, s in enumerate(tile_sizes_v) if s != 0}
        tile_sizes = {
            i: arith.Constant(IntegerAttr.from_index_int_value(s))
            for i, s in tile_sizes_v.items()
        }
        tiled_dims = sorted(tile_sizes.keys())
        outter_lower = [lower[d] for d in tiled_dims]
        outter_upper = [upper[d] for d in tiled_dims]
        outter_step = [arith.Muli(step[d], tile_sizes[d]) for d in tiled_dims]

        outter_loop = ParallelOp(
            outter_lower,
            outter_upper,
            outter_step,
            Region(
                Block(
                    [(outter_yield := Yield())],
                    arg_types=[IndexType()] * len(outter_lower),
                )
            ),
        )

        inner_lower = list[Operation]()
        inner_upper = list[Operation]()
        minops = list[Operation]()
        minmap = affine.AffineMapAttr(
            affine.AffineMap(
                3,
                0,
                (
                    affine.AffineExpr.dimension(0),
                    affine.AffineExpr.dimension(1) - affine.AffineExpr.dimension(2),
                ),
            )
        )
        for i in range(len(op.lowerBound)):
            if i in tile_sizes:
                inner_lower.append(zero)
                iter_count = (upper_v[i] - lower_v[i]) // step_v[i]
                if iter_count % tile_sizes_v[i] == 0:
                    inner_upper.append(tile_sizes[i])
                else:
                    arg_index = tiled_dims.index(i)
                    minop = affine.MinOp(
                        operands=[
                            [
                                tile_sizes[i],
                                outter_upper[i],
                                outter_loop.body.block.args[arg_index],
                            ]
                        ],
                        properties={"map": minmap},
                        result_types=[IndexType()],
                    )
                    minops.append(minop)
                    inner_upper.append(minop)

            else:
                inner_lower.append(lower[i])
                inner_upper.append(upper[i])

        inner_loop = ParallelOp(inner_lower, inner_upper, step, op.detach_region(0))
        for i, arg in reversed(list(enumerate(inner_loop.body.block.args))):
            if i in tile_sizes:
                arg_index = tiled_dims.index(i)
                iv = arith.Addi(outter_loop.body.block.args[arg_index], arg)
                assert inner_loop.body.block.first_op is not None
                inner_loop.body.block.insert_op_before(
                    iv, inner_loop.body.block.first_op
                )
                for use in arg.uses:
                    if use.operation is iv:
                        continue
                    use.operation.operands[use.index] = iv.result
        outter_loop.body.block.insert_ops_before([*minops, inner_loop], outter_yield)
        # print("tiled")
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
