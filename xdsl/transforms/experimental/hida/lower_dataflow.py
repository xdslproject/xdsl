from dataclasses import dataclass

from xdsl.builder import InsertPoint
from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, func
from xdsl.dialects.experimental.hida_functional import DispatchOp, TaskOp
from xdsl.dialects.experimental.hida_structural import NodeOp, ScheduleOp
from xdsl.dialects.experimental.utils import is_written
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Block, BlockArgument, Region, SSAValue, Use
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.experimental.liveness import Liveness
from xdsl.transforms.lower_affine import LowerAffineLoad, LowerAffineStore
from xdsl.utils.hints import isa


@dataclass
class LowerDispatchToSchedule(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, dispatch: DispatchOp, rewriter: PatternRewriter):
        if dispatch.results:
            return dispatch.emit_error("should not yield any results")

        def is_in_dispatch(use: Use):
            return dispatch.is_ancestor(use.operation)

        inputs: list[SSAValue] = []
        liveins = Liveness(dispatch).get_livein(dispatch.region.block)

        for livein in liveins:
            parent_region = livein.owner.parent_region()
            assert parent_region
            if dispatch.region.is_ancestor(parent_region):
                continue

            parent_region = livein.owner.parent_region()
            assert isinstance(parent_region, Region)
            if dispatch.region.is_ancestor(parent_region):
                continue
            inputs.append(livein)

        schedule_block = Block()
        for input in reversed(inputs):
            rewriter.insert_block_argument(schedule_block, 0, input.type)

        for t in zip(inputs, schedule_block.args):
            t[0].replace_by_if(t[1], is_in_dispatch)

        for dispatch_op in dispatch.region.block.ops:
            if dispatch_op != dispatch.region.block.last_op:
                dispatch_op.detach()
                schedule_block.add_op(dispatch_op)
            else:
                dispatch_op.detach()

        schedule = ScheduleOp(schedule_block, inputs)

        rewriter.insert_op(schedule, InsertPoint.before(dispatch))
        rewriter.erase_op(dispatch)


@dataclass
class LowerTaskToNode(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, task: TaskOp, rewriter: PatternRewriter):
        if task.results:
            return task.emit_error("should not yield any results")

        def is_in_task(use: Use):
            return task.is_ancestor(use.operation)

        inputs: list[SSAValue] = []
        outputs: list[SSAValue] = []
        params: list[SSAValue] = []

        liveins = Liveness(task).get_livein(task.region.block)

        for livein in liveins:
            parent_region = livein.owner.parent_region()
            assert parent_region
            if task.region.is_ancestor(parent_region):
                continue

            if isa(
                livein.type, MemRefType
            ):  # TODO: or stream type (not yet implemented)
                uses = list(filter(lambda x: is_in_task(x), livein.uses))
                if any(map(lambda x: is_written(x), uses)):
                    outputs.append(livein)
                else:
                    inputs.append(livein)
            else:
                params.append(livein)

        node_block = Block()

        input_args: list[BlockArgument] = []
        for input in inputs:
            rewriter.insert_block_argument(node_block, len(node_block.args), input.type)
            input_args.append(node_block.args[-1])

        for t in zip(inputs, input_args):
            t[0].replace_by_if(t[1], is_in_task)

        output_args: list[BlockArgument] = []
        for output in outputs:
            rewriter.insert_block_argument(
                node_block, len(node_block.args), output.type
            )
            output_args.append(node_block.args[-1])
        for t in zip(outputs, output_args):
            t[0].replace_by_if(t[1], is_in_task)

        param_args: list[BlockArgument] = []
        for param in params:
            rewriter.insert_block_argument(node_block, len(node_block.args), param.type)
            param_args.append(node_block.args[-1])
        for t in zip(params, param_args):
            t[0].replace_by_if(t[1], is_in_task)

        for task_op in task.region.block.ops:
            if task_op != task.region.block.last_op:
                task_op.detach()
                node_block.add_op(task_op)
            else:
                task_op.detach()

        node = NodeOp(node_block, inputs + outputs + params)

        rewriter.insert_op(node, InsertPoint.before(task))
        rewriter.erase_op(task)


@dataclass
class LocalizeConstants(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter):
        constants: list[arith.Constant] = []
        for op in filter(lambda x: isinstance(x, arith.Constant), func_op.walk()):
            assert isinstance(op, arith.Constant)
            constants.append(op)

        for constant in constants:
            for use in constant.result.uses.copy():
                clone_constant = constant.clone()
                rewriter.insert_op(clone_constant, InsertPoint.before(use.operation))
                use.operation.operands[use.index] = clone_constant.result

            constant.detach()
            constant.erase()


@dataclass(frozen=True)
class LowerDataflow(ModulePass):
    name = "hida-lower-dataflow"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        localize_constants = PatternRewriteWalker(
            GreedyRewritePatternApplier([LocalizeConstants()]), apply_recursively=False
        )
        localize_constants.rewrite_module(op)

        lower_dispatch = PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerDispatchToSchedule(), LowerTaskToNode()]),
        )
        lower_dispatch.rewrite_module(op)

        lower_affine_mem = PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerAffineLoad(), LowerAffineStore()])
        )
        lower_affine_mem.rewrite_module(op)
