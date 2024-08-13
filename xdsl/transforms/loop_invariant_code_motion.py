from collections.abc import Iterator
from typing import Any, Sequence, cast
import re
import queue
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, scf, memref
from xdsl.ir import SSAValue, Operation, Block, Region
from xdsl.traits import IsTerminator, NoTerminator, OpTrait, OpTraitInvT
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

#  This pass flattens pairs nested loops into a single loop.
#
#  Similar to LLVM's loop flattening:
#  https://llvm.org/doxygen/LoopFlatten_8cpp_source.html
#
#  An operation is loop-invariant if it depends only of values defined outside of the loop. LICM moves these operations out of the loop body so that they are not computed more than once.
#
#    for i in range(x, N, M):
#      for j in range(0, M, K):
#        f(A[i+j])
#
#    for o in range(ol, ou, os):
#      for i in range(il, iu, is):
#        # neither o nor i are used
#
#    These become:
#    # (If K is constant and divides M)
#    for i in range(x, N, K):
#      f(A[i])
#
#    factor = (iu - il) // is
#    for o in range(ol, ou * factor, os):
#      # o is not used
#

#  Checks whether the given op can be hoisted by checking that
#  - the op and none of its contained operations depend on values inside of the
#    loop (by means of calling definedOutside).
#  - the op has no side-effects.
def canBeHoisted(op: Operation, condition: Callable[[SSAValue], bool]) -> bool | None:
    #   Do not move terminators.
    if op.has_trait(IsTerminator):
        return False

    # Walk the nested operations and check that all used values are either
    # defined outside of the loop or in a nested region, but not at the level of
    # the loop body.

    return any(
        op.is_ancestor(operand.owner) or condition(operand)
        for child in op.walk()
        for operand in child.operands
    )


def isDefinedOutsideOfRegoin(value: SSAValue, region: Region) -> bool | None:
    return not region.is_ancestor(value.owner)

# def can_be_hoisted_with_value_check(op: Operation, defined_outside: Callable[[SSAValue], bool])-> bool | None:
#     return canBeHoisted(op, defined_outside(op))

# def moveLoopInvariantCode(
#     regions : Sequence[Region],
#     isDefinedOutsideRegion: Callable[[Region], bool], #function_ref<bool(Value, Region *)> isDefinedOutsideRegion
#     shouldMoveOutofRegion: Callable[[Operation], bool], #function_ref<bool(Operation *, Region *)> shouldMoveOutOfRegion
#     moveOutofRegion: Callable[[Operation], Region]) -> int | None: #function_ref<void(Operation *, Region *)> moveOutOfRegion
    
#     numMoved = 0
#     worklist : list[Operation] = []

#     for region in regions: #iter thorugh the regions
#         print("Original loop: ", region.parent_node)
#         for op in region.block.ops:
#             print("Operation: ", op)
#             worklist.append(op)
        
#             definedOutside = isDefinedOutsideRegion(region)

#             while not worklist:
#                 oper = worklist.pop()
#                 #Skip ops that have already been moved. Check if the op can be hoisted.
#                 if oper.parent_region() != region:
#                     continue
#                 print("Check op: ", oper)

#                 if not(shouldMoveOutofRegion(oper, region) or not(canBeHoisted(oper, definedOutside))):
#                     continue
#                 print("Moving loop-invariant op: ", oper)
#                 moveOutofRegion(oper, region)
#                 numMoved = numMoved + 1

#                 for user in oper.results[0].uses:
#                     if user.operation.parent_region is region:
#                         worklist.append(user.operation)

#     return numMoved

class LoopsInvariantCodeMotion(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter) -> None:
        for region in op.regions:
            for block in region.blocks:
                for lp in block.walk():
                    for oper in lp.operands:
                        print("lp: ",lp)
                        print("region: ", region)
                        print("operands: ", oper._name)
                        print("isDefinedOutside: ", isDefinedOutsideOfRegoin(oper, region))


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
