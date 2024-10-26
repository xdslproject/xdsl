from dataclasses import dataclass
from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.pattern_rewriter import PatternRewriteWalker, GreedyRewritePatternApplier, RewritePattern, op_type_rewrite_pattern, PatternRewriter
from xdsl.irdl import Operation
from xdsl.dialects.experimental.utils import dispatch_block, fuse_ops_into_task
from xdsl.dialects import func, linalg, tensor
from xdsl.dialects.experimental.hida_functional import TaskOp
from xdsl.utils.hints import isa

def backward_fuse(op : Operation, rewriter : PatternRewriter):
    if isinstance(op.parent_op(), TaskOp):
        return

    task_def_ops = list(map(lambda x: x.owner, filter(lambda x: isinstance(x.owner, TaskOp), op.operands)))
    assert isa(task_def_ops, list[Operation])

    fuse_ops_into_task(task_def_ops + [op], rewriter, True)

def forward_fuse(op : Operation, rewriter : PatternRewriter):
    if isinstance(op.parent_op(), TaskOp):
        return

    all_task_users = list(filter(lambda x: isinstance(x, TaskOp), [use.operation.parent_op() for result in op.results for use in result.uses]))
    assert isa(all_task_users, list[Operation])

    fuse_ops_into_task([op] + all_task_users, rewriter, False)

@dataclass 
class ForwardFuseLinalgGeneric(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op : linalg.Generic, rewriter : PatternRewriter, /):
        forward_fuse(op, rewriter)

@dataclass
class ForwardFuseLinalgFillOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op : linalg.FillOp, rewriter : PatternRewriter, /):
        forward_fuse(op, rewriter)

@dataclass
class ForwardFuseTensorEmptyOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op : tensor.EmptyOp, rewriter : PatternRewriter, /):
        forward_fuse(op, rewriter)

@dataclass
class ForwardFuseTensorPadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op : tensor.PadOp, rewriter : PatternRewriter, /):
        forward_fuse(op, rewriter)

@dataclass
class ForwardFuseTensorCollapseShapeOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op : tensor.CollapseShapeOp, rewriter : PatternRewriter, /):
        forward_fuse(op, rewriter)

# TODO: Operation not yet implement in xDSL
#@dataclass
#class ForwardFuseExpandShapeOp(RewritePattern):
#    @op_type_rewrite_pattern
#    def match_and_rewrite(self, op : tensor.ExpandShapeOp, rewriter : PatternRewriter, /):
#        forward_fuse(op, rewriter)

@dataclass
class ForwardFuseTensorInsertSliceOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op : tensor.InsertSliceOp, rewriter : PatternRewriter, /):
        forward_fuse(op, rewriter)

@dataclass
class ForwardFuseTensorExtractSliceOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op : tensor.ExtractSliceOp, rewriter : PatternRewriter, /):
        forward_fuse(op, rewriter)


@dataclass 
class BackwardFuseLinalgGeneric(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op : linalg.Generic, rewriter : PatternRewriter, /):
        backward_fuse(op, rewriter)

@dataclass
class DispatchBlocks(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op : func.FuncOp, rewriter : PatternRewriter, /):
        dispatch_block(op.body.block)
        #assert isinstance(op.body.block.last_op, Operation)
        #op.body.block.insert_op_before(dispatch_op, op.body.block.last_op)


@dataclass(frozen=True)
class CreateDataflowFromLinalg(ModulePass):
    name = "hida-create-dataflow-from-linalg"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:

        inout_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    DispatchBlocks(),
                    BackwardFuseLinalgGeneric(),
                    ForwardFuseLinalgGeneric(),
                    ForwardFuseLinalgFillOp(),
                    ForwardFuseTensorEmptyOp(),
                    ForwardFuseTensorPadOp(),
                    ForwardFuseTensorCollapseShapeOp(),
                    #ForwardFuseTensorExpandShapeOp(),
                    ForwardFuseTensorInsertSliceOp(),
                    ForwardFuseTensorExtractSliceOp()
                ]
            ),
            apply_recursively=False,
            walk_reverse=False,
        )
        inout_pass.rewrite_module(op)