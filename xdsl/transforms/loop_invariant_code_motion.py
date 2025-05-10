"""
This pass hoists operation that are invariant to the loops.

Similar to MLIR's loop invariant code motion: see external [documentation](https://mlir.llvm.org/doxygen/LoopInvariantCodeMotion_8cpp_source.html).

An operation is loop-invariant if it depends only of values defined outside of the loop.
LICM moves these operations out of the loop body so that they are not computed more than
once.

  for i in range(x, N, M):                for i in range(x, N, M):
    for j in range(0, M, K):    ---->        c[i]= A[1] + b[1]
      c[i]=A[1]+b[1]
"""

from typing import Callable, Sequence
from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import builtin, scf
from xdsl.ir import Operation, Region
from xdsl.ir.core import SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    Worklist,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.traits import IsTerminator, is_side_effect_free, is_speculatable


def can_be_hoisted(op: Operation, target_region: Region) -> bool | None:
    """
    Checks whether the given op can be hoisted by checking that
    - the op and none of its contained operations depend on values inside of the
     loop.
    """
    #  Do not move terminators.
    if op.has_trait(IsTerminator):
        return False

    # Walk the nested operations and check that all used values are either
    # defined outside of the loop or in a nested region, but not at the level of
    # the loop body.
    for child in op.walk():
        for operand in child.operands:
            operand_owner = operand.owner
            if op.is_ancestor(operand_owner):
                continue
            if target_region.is_ancestor(operand_owner):
                return False
    return True


def _move_loop_invariant_code(
    regions: Sequence[Region],
    should_move_out_of_region: Callable[[Operation, Region], bool],
    move_out_of_region: Callable[[Operation, Region], None],
):
    num_moved = 0
    for region in regions:
        # add top-level operations in the loop body to the worklist
        worklist = [op for block in region.blocks for op in block.ops]

        while worklist:
            op = worklist.pop(0)
            # Skip ops that have already been moved. Check if the op can be hoisted.
            if op.parent_region() != region:
                continue

            if not (
                should_move_out_of_region(op, region) and can_be_hoisted(op, region)
            ):
                continue

            move_out_of_region(op, region)
            num_moved += 1

            # Since the op has been moved, we need to check its users within the
            # top-level of the loop body.

            for res in op.results:
                for use in res.uses:
                    user = use.operation
                    if user.parent_region() == region:
                        worklist.append(user)

    return num_moved


def _should_move_out_of_region(op: Operation, region: Region) -> bool:
    return is_side_effect_free(op) and is_speculatable(op)


def move_loop_invariant_code(loop: scf.ForOp):
    loop_regions = tuple(op.body for op in loop.walk() if isinstance(op, scf.ForOp))

    builder = Builder(InsertPoint.before(loop))

    def _move_out_of_region(op: Operation, region: Region) -> None:
        op.detach()
        builder.insert(op)

    _move_loop_invariant_code(
        loop_regions, _should_move_out_of_region, _move_out_of_region
    )


class LoopsInvariantCodeMotion(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter) -> None:
        move_loop_invariant_code(op)


class LoopInvariantCodeMotionPass(ModulePass):
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

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            LoopsInvariantCodeMotion(), walk_reverse=True
        ).rewrite_module(op)
