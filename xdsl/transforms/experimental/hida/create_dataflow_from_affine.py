from dataclasses import dataclass

from xdsl.builder import InsertPoint
from xdsl.context import MLContext
from xdsl.dialects import affine, arith, builtin, func, memref, scf
from xdsl.dialects.experimental.hida_functional import DispatchOp, TaskOp
from xdsl.dialects.experimental.hida_structural import NodeOp
from xdsl.dialects.experimental.utils import (
    dispatch_block,
    fuse_ops_into_task,
    get_loop_bands,
)
from xdsl.ir import Block, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import IsTerminator
from xdsl.utils.hints import isa


@dataclass
class TaskPartition(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, dispatch: DispatchOp, rewriter: PatternRewriter):
        for op in dispatch.region.block.ops:
            if any(
                map(
                    lambda x: x == op.dialect_name(),
                    ["bufferization", "tosa", "tensor", "linalg"],
                )
            ) or any(
                map(
                    lambda x: isa(op, x),
                    [func.Call, DispatchOp, TaskOp, DispatchOp, NodeOp],
                )
            ):
                return

        block: Block = dispatch.region.block

        ops_to_fuse: list[Operation] = []

        task_idx = 0

        for op in block.ops:
            # TODO: Check for memory effects from the operation, not just memref.Alloc
            if isinstance(op, memref.Alloc):
                assert isinstance(block.first_op, Operation)
                op.detach()
                rewriter.insert_op(op, InsertPoint.before(block.first_op))

            elif any(map(lambda x: isinstance(op, x), [affine.For, scf.For])):
                ops_to_fuse.append(op)
                fuse_ops_into_task(ops_to_fuse, rewriter)
                ops_to_fuse = []
                task_idx += 1

            elif op == block.last_op and op.has_trait(IsTerminator):
                if not ops_to_fuse or task_idx == 0:
                    continue
                fuse_ops_into_task(ops_to_fuse, rewriter)
                ops_to_fuse = []
                task_idx += 1
            else:
                ops_to_fuse.append(op)


@dataclass
class DispatchBlocks(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        dispatch_block(op.body.block)
        # assert isinstance(op.body.block.last_op, Operation)
        # op.body.block.insert_op_before(dispatch_op, op.body.block.last_op)

        target_bands: list[list[affine.For]] = []
        get_loop_bands(op.body.block, target_bands, True)
        for band in reversed(target_bands):
            dispatch_block(band[-1].body.block)

            # assert isinstance(band[-1].body.block.last_op, Operation)
            # rewriter.insert_op(dispatch_op, InsertPoint.before(band[-1].body.block.last_op))


# Temporary fix for the lack of constant folding in xDSL. MLIR applies constant folding before applying other patterns,
# as a result, some constants might not belong in the dispatch regions.
@dataclass
class ConstantFolding(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Constant, rewriter: PatternRewriter):
        if len(op.result.uses) > 2:
            parent_op = op.parent_op()
            while not isinstance(parent_op, func.FuncOp):
                parent_op = parent_op.parent_op()

            op.detach()
            rewriter.insert_op(op, InsertPoint.at_start(parent_op.body.block))


@dataclass(frozen=True)
class CreateDataflowFromAffine(ModulePass):
    name = "hida-create-dataflow-from-affine"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        dispatch_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    DispatchBlocks(),
                ]
            ),
            apply_recursively=False,
            walk_reverse=False,
        )
        dispatch_pass.rewrite_module(op)

        constant_folding_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConstantFolding(),
                ]
            ),
            apply_recursively=False,
            walk_reverse=False,
        )
        constant_folding_pass.rewrite_module(op)

        task_partition_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    TaskPartition(),
                ]
            ),
            apply_recursively=False,
            walk_reverse=False,
        )
        task_partition_pass.rewrite_module(op)
