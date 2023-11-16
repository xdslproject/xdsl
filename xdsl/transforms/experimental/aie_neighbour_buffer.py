from dataclasses import dataclass

from xdsl.builder import Builder
from xdsl.dialects import builtin
from xdsl.dialects.experimental import aie
from xdsl.ir import (
    MLContext,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class NorthNeighbour(RewritePattern):
    module: builtin.ModuleOp

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aie.CoreOp, rewriter: PatternRewriter, /):
        print("FOUND CORE: ", op.tile.op.row)

        device = None
        for _op in self.module.ops:
            if isinstance(_op, aie.DeviceOp):
                device = _op

        print(device)
        # Instantiate north neighbour tile and core
        row = builtin.IntegerAttr.from_int_and_width(
            op.tile.op.row.value.data + 1, builtin.i32
        )
        col = op.tile.op.col

        north_tile = aie.TileOp(row, col)

        @Builder.region
        def core_region(builder: Builder):
            builder.insert(aie.EndOp())

        north_core = aie.CoreOp(
            builtin.IntegerAttr.from_int_and_width(1, builtin.i32),
            north_tile,
            core_region,
        )

        print(north_core)

        rewriter.insert_op_after_matched_op([north_tile, north_core])


@dataclass
class AIENeighbourBuffer(ModulePass):
    name = "aie-neighbour-buffer"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        north_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier([NorthNeighbour(op)]),
            apply_recursively=False,
            walk_reverse=False,
        )
        north_pass.rewrite_module(op)
