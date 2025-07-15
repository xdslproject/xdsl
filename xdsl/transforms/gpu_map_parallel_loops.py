from xdsl.context import Context
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, ModuleOp
from xdsl.dialects.gpu import LoopDimMapAttr, ProcessorAttr, ProcessorEnum
from xdsl.dialects.scf import ParallelOp
from xdsl.ir import Operation
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

MappingAttrName = "mapping"

MapGrid = 0
MapBlock = 1
Sequential = 2

kNumHardwareIds = 3


def getHardwareIdForMapping(level: int, dimension: int) -> ProcessorEnum:
    """
    Computed the hardware id to use for a given mapping level. Will
    assign x,y and z hardware ids for the first 3 dimensions and use
    sequential after.
    """

    if dimension >= kNumHardwareIds or level == Sequential:
        return ProcessorEnum.Sequential
    match level:
        case 0:
            match dimension:
                case 0:
                    return ProcessorEnum.Block_X
                case 1:
                    return ProcessorEnum.Block_Y
                case 2:
                    return ProcessorEnum.Block_Z
                case _:
                    return ProcessorEnum.Sequential
        case 1:
            match dimension:
                case 0:
                    return ProcessorEnum.Thread_X
                case 1:
                    return ProcessorEnum.Thread_Y
                case 2:
                    return ProcessorEnum.Thread_Z
                case _:
                    return ProcessorEnum.Sequential
        case _:
            return ProcessorEnum.Sequential


def mapParallelOp(parallelOp: ParallelOp, mappingLevel: int = MapGrid):
    """
    Add mapping information to the given parallel loop. Do not add
    mapping information if the loop already has it. Also, don't
    start a mapping at a nested loop.
    """
    # Do not try to add a mapping to already mapped loops or nested loops.
    anchor: Operation | None = parallelOp.parent_op()
    while (anchor is not None) and (not isinstance(anchor, ParallelOp)):
        anchor = anchor.parent_op()

    if (MappingAttrName in parallelOp.attributes) or (
        (mappingLevel == MapGrid) and anchor is not None
    ):
        return
    attrs = [
        getHardwareIdForMapping(mappingLevel, i)
        for i in range(len(parallelOp.lowerBound))
    ]
    attrs = ArrayAttr(
        [
            LoopDimMapAttr(
                ProcessorAttr(attr),
                AffineMapAttr(AffineMap.identity(1)),
                AffineMapAttr(AffineMap.identity(1)),
            )
            for attr in reversed(attrs)
        ]
    )
    parallelOp.attributes[MappingAttrName] = attrs
    mappingLevel += 1
    for op in parallelOp.body.ops:
        if isinstance(op, ParallelOp):
            mapParallelOp(op, mappingLevel)


class GpuMapParallelLoopsPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ParallelOp, rewriter: PatternRewriter, /):
        mapParallelOp(op)


class GpuMapParallelLoopsPass(ModulePass):
    name = "gpu-map-parallel-loops"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([GpuMapParallelLoopsPattern()])
        )
        walker.rewrite_module(op)
