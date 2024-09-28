from xdsl.context import MLContext
from xdsl.dialects import builtin, scf
from xdsl.ir import Operation, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import (
    IsTerminator,
    # is_side_effect_free,
    # only_has_effect,
)

#  This pass hoists operation that are invariant to the loops.
#
#  Similar to MLIR's loop invariant code motion:
#  https://mlir.llvm.org/doxygen/LoopInvariantCodeMotion_8cpp_source.html
#
#  An operation is loop-invariant if it depends only of values defined outside of the loop. LICM moves these operations out of the loop body so that they are not computed more than once.
#
#    for i in range(x, N, M):                for i in range(x, N, M):
#      for j in range(0, M, K):    ---->        c[i]= A[1] + b[1]
#        c[i]=A[1]+b[1]


#  Checks whether the given op can be hoisted by checking that
#  - the op and none of its contained operations depend on values inside of the
#    loop (by means of calling definedOutside).
#  - the op has no side-effects.
def canBeHoisted(op: Operation, region_target: Region) -> bool | None:
    #   Do not move terminators.
    if op.has_trait(IsTerminator):
        return False

    # Walk the nested operations and check that all used values are either
    # defined outside of the loop or in a nested region, but not at the level of
    # the loop body.
    for child in op.walk():
        for operand in child.operands:
            for own in operand.owner.walk():
                if not isinstance(own, scf.For):
                    if op.is_ancestor(operand.owner):
                        continue
                    if region_target.is_ancestor(own):
                        return False
    return True


class LoopsInvariantCodeMotion(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter) -> None:

        if any(isinstance(ha, scf.For) for ha in op.body.walk()):
            return

        numMoved = 0

        worklist: list[Operation] = []

        for region in op.regions:  # iter thorugh the regions
            for ops in region.block.ops:
                worklist.append(ops)
                while worklist:
                    oper = worklist.pop()
                    # Skip ops that have already been moved. Check if the op can be hoisted.
                    if oper.parent_region() != region:
                        continue
                    if not canBeHoisted(oper, region):
                        continue
                    print("Can be hoisted op: ", oper)

                    numMoved = numMoved + 1
                    if not isinstance(oper, scf.Yield):
                        for user in oper.results[0].uses:
                            if user.operation.parent_region is region:
                                worklist.append(user.operation)

        # print(numMoved)


class ScfForLoopInavarintCodeMotionPass(ModulePass):
    """
    Folds perfect loop nests if they can be represented with a single loop.
    Currently does this by matching the inner loop range with the outer loop step.
    If the inner iteration space fits perfectly in the outer iteration step, then merge.
    Other conditions:
     - the only use of the induction arguments must be an add operation, this op is fused
       into a single induction argument,
     - the lower bound of the inner loop must be 0,
     - the loops must have no iteration arguments.
    """

    name = "licm"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(LoopsInvariantCodeMotion()).rewrite_module(op)
