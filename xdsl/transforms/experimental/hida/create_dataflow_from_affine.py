from dataclasses import dataclass
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, GreedyRewritePatternApplier, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.dialects import builtin
from xdsl.context import MLContext
from xdsl.dialects.experimental.hida_functional import DispatchOp, TaskOp

@dataclass
class TaskPartition(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op : DispatchOp, rewriter: PatternRewriter):
        print("DIALECT NAME: ", op.dialect_name())
    
    

@dataclass(frozen=True)
class CreateDataflowFromAffine(ModulePass):
    name = "hida-create-dataflow-from-affine"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:

        inout_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    TaskPartition()
                ]
            ),
            apply_recursively=False,
            walk_reverse=False,
        )
        inout_pass.rewrite_module(op)