from dataclasses import dataclass

from xdsl.builder import InsertPoint
from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, func, linalg, tensor
from xdsl.dialects.experimental.hida_functional import TaskOp
from xdsl.dialects.experimental.utils import (
    dispatch_block,
    fuse_ops_into_task,
    is_element_wise_generic_op,
)
from xdsl.irdl import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import IsContraction
from xdsl.utils.hints import isa


def backward_fuse(op: Operation, rewriter: PatternRewriter):
    if isinstance(op.parent_op(), TaskOp):
        return

    task_def_ops = list(
        map(
            lambda x: x.owner,
            filter(lambda x: isinstance(x.owner, TaskOp), op.operands),
        )
    )
    assert isa(task_def_ops, list[Operation])

    if not task_def_ops:
        return

    # TODO: dominance analysis to sort the dependencies
    fuse_ops_into_task([task_def_ops[-1]] + [op], rewriter, True)


def forward_fuse(op: Operation, rewriter: PatternRewriter):
    if isinstance(op.parent_op(), TaskOp):
        return

    all_task_users = list(
        filter(
            lambda x: isinstance(x, TaskOp),
            [use.operation.parent_op() for result in op.results for use in result.uses],
        )
    )
    assert isa(all_task_users, list[TaskOp])

    if not all_task_users:
        return

    # TODO: dominance analysis to sort the dependencies
    fuse_ops_into_task([op] + all_task_users, rewriter, True)


@dataclass
class OutlineContraction(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if op.has_trait(IsContraction):
            if op.get_parent_of_type(TaskOp):
                return

            fuse_ops_into_task([op], rewriter)


@dataclass
class OutlineConvolution(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.ConvOpsBase | linalg.PoolingOpsBase, rewriter: PatternRewriter
    ):
        if op.get_parent_of_type(TaskOp):
            return

        fuse_ops_into_task([op], rewriter)


@dataclass
class OutlineLinalgOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        if op.get_parent_of_type(TaskOp):
            return

        fuse_ops_into_task([op], rewriter)


@dataclass
class ForwardFuseLinalgGeneric(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter, /):
        matched = False

        if len(op.outputs) == 1 and len(op.body.block.ops) == 1:
            output = list(op.body.block.get_terminator().operands)[0]

            if len(op.inputs) == 1 and output == op.body.block.args[0]:
                matched = True

            if isinstance(
                output.owner, arith.Constant
            ):  # TODO: or isinstance(output.owner, tosa.Constant)
                matched = True

        if matched:
            forward_fuse(op, rewriter)


@dataclass
class ForwardFuseLinalgFillOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.FillOp, rewriter: PatternRewriter, /):
        forward_fuse(op, rewriter)


@dataclass
class ForwardFuseTensorEmptyOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tensor.EmptyOp, rewriter: PatternRewriter, /):
        forward_fuse(op, rewriter)


@dataclass
class ForwardFuseTensorPadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tensor.PadOp, rewriter: PatternRewriter, /):
        forward_fuse(op, rewriter)


@dataclass
class ForwardFuseTensorCollapseShapeOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: tensor.CollapseShapeOp, rewriter: PatternRewriter, /
    ):
        forward_fuse(op, rewriter)


# TODO: Operation not yet implement in xDSL
# @dataclass
# class ForwardFuseExpandShapeOp(RewritePattern):
#    @op_type_rewrite_pattern
#    def match_and_rewrite(self, op : tensor.ExpandShapeOp, rewriter : PatternRewriter, /):
#        forward_fuse(op, rewriter)


@dataclass
class ForwardFuseTensorInsertSliceOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tensor.InsertSliceOp, rewriter: PatternRewriter, /):
        forward_fuse(op, rewriter)


@dataclass
class ForwardFuseTensorExtractSliceOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: tensor.ExtractSliceOp, rewriter: PatternRewriter, /
    ):
        forward_fuse(op, rewriter)


@dataclass
class BackwardFuseLinalgGeneric(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter, /):
        if is_element_wise_generic_op(op):
            backward_fuse(op, rewriter)


@dataclass
class DispatchBlocks(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        dispatch_block(op.body.block)

        empty_tensors = [
            empty
            for empty in filter(lambda x: isinstance(x, tensor.EmptyOp), op.walk())
        ]

        for empty in empty_tensors:
            assert isinstance(empty, tensor.EmptyOp)
            for use in empty.tensor.uses.copy():
                clone_empty = empty.clone()
                use.operation.operands[use.index] = clone_empty.tensor
                rewriter.insert_op(clone_empty, InsertPoint.before(use.operation))

            empty.detach()
            empty.erase()


fuse_pass_calls = [
    BackwardFuseLinalgGeneric(),
    ForwardFuseLinalgGeneric(),
    ForwardFuseLinalgFillOp(),
    ForwardFuseTensorEmptyOp(),
    ForwardFuseTensorPadOp(),
    ForwardFuseTensorCollapseShapeOp(),
    # ForwardFuseTensorExpandShapeOp(),
    ForwardFuseTensorInsertSliceOp(),
    ForwardFuseTensorExtractSliceOp(),
]


@dataclass(frozen=True)
class CreateDataflowFromLinalg(ModulePass):
    name = "hida-create-dataflow-from-linalg"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        dispatch_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [DispatchBlocks()],
            ),
            apply_recursively=False,
        )
        dispatch_pass.rewrite_module(op)

        fuse_pass_1 = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [OutlineConvolution(), OutlineContraction()] + fuse_pass_calls
            ),
        )
        fuse_pass_1.rewrite_module(op)

        fuse_pass_2 = PatternRewriteWalker(
            GreedyRewritePatternApplier([OutlineLinalgOp()] + fuse_pass_calls),
            walk_reverse=True,
        )
        fuse_pass_2.rewrite_module(op)
