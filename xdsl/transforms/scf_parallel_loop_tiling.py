from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import affine, arith
from xdsl.dialects.builtin import IndexType, IntegerAttr, ModuleOp
from xdsl.dialects.scf import ParallelOp, ReduceOp
from xdsl.ir import Block, Operation, Region, SSAValue
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
    tile_sizes: tuple[int, ...]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ParallelOp, rewriter: PatternRewriter, /):
        # Only tile the innermost parallel loop
        if any(isinstance(o, ParallelOp) for o in op.body.walk()):
            return
        lower = op.lowerBound
        upper = op.upperBound
        step = op.step
        # The pass is meant to work on any parallel loop with any nu;ber of tile sizes.
        # For a loop of dimension N, either use the N first tile sizes or use them all
        # and fill the rest with 1.
        tile_sizes_v = self.tile_sizes[: len(lower)] + (1,) * (
            len(lower) - len(self.tile_sizes)
        )

        zero = arith.ConstantOp(IntegerAttr.from_index_int_value(0))
        tile_sizes_v = {i: s for i, s in enumerate(tile_sizes_v) if s != 0}
        tile_sizes = {
            i: arith.ConstantOp(IntegerAttr.from_index_int_value(s))
            for i, s in tile_sizes_v.items()
        }
        tiled_dims = sorted(tile_sizes.keys())
        if not tiled_dims:
            return
        outter_lower = [lower[d] for d in tiled_dims]
        outter_upper = [upper[d] for d in tiled_dims]
        outter_step = [arith.MuliOp(step[d], tile_sizes[d]) for d in tiled_dims]

        outter_loop = ParallelOp(
            outter_lower,
            outter_upper,
            outter_step,
            Region(
                Block(
                    [(outter_reduce := ReduceOp())],
                    arg_types=[IndexType()] * len(outter_lower),
                )
            ),
        )

        inner_lower = list[SSAValue | Operation]()
        inner_upper = list[SSAValue | Operation]()
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
                ilower, iupper, istep = lower[i], upper[i], step[i]
                if (
                    isinstance(ilower, arith.ConstantOp)
                    and isinstance(iupper, arith.ConstantOp)
                    and isinstance(istep, arith.ConstantOp)
                ):
                    lower_v, upper_v, step_v = (
                        c.value for c in (ilower, iupper, istep)
                    )
                    assert isa(lower_v, IntegerAttr[IndexType])
                    assert isa(upper_v, IntegerAttr[IndexType])
                    assert isa(step_v, IntegerAttr[IndexType])
                    lower_v, upper_v, step_v = (
                        c.value.data for c in (lower_v, upper_v, step_v)
                    )
                    iter_count = (upper_v - lower_v) // step_v
                    if iter_count % tile_sizes_v[i] == 0:
                        inner_upper.append(tile_sizes[i])
                        continue

                arg_index = tiled_dims.index(i)
                minop = affine.MinOp(
                    operands=[
                        [
                            tile_sizes[i],
                            outter_upper[arg_index],
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
                iv = arith.AddiOp(outter_loop.body.block.args[arg_index], arg)
                assert inner_loop.body.block.first_op is not None
                inner_loop.body.block.insert_op_before(
                    iv, inner_loop.body.block.first_op
                )
                for use in tuple(arg.uses):
                    if use.operation is iv:
                        continue
                    use.operation.operands[use.index] = iv.result
        outter_loop.body.block.insert_ops_before([*minops, inner_loop], outter_reduce)
        rewriter.replace_op(op, [zero, *tile_sizes.values(), *outter_step, outter_loop])


@dataclass(frozen=True)
class ScfParallelLoopTilingPass(ModulePass):
    name = "scf-parallel-loop-tiling"

    parallel_loop_tile_sizes: tuple[int, ...]

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [ScfParallelLoopTilingPattern(tuple(self.parallel_loop_tile_sizes))]
            ),
            walk_regions_first=True,
            apply_recursively=False,
        )
        walker.rewrite_module(op)
