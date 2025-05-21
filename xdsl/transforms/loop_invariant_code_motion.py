"""
This pass hoists operation that are invariant to the loops.

Similar to MLIR's loop invariant code motion: see external [documentation](https://mlir.llvm.org/doxygen/LoopInvariantCodeMotion_8cpp_source.html).

An operation is loop-invariant if it depends only of values defined outside of the loop.
LICM moves these operations out of the loop body so that they are not computed more than
once.
"""

from collections import deque

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import builtin, scf
from xdsl.ir import Operation, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
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


def _move_loop_invariant_code(region: Region, builder: Builder):
    # add top-level operations in the loop body to the worklist
    worklist = deque(op for block in region.blocks for op in block.ops)

    while worklist:
        op = worklist.popleft()
        # Skip ops that have already been moved. Check if the op can be hoisted.
        if op.parent_region() != region:
            continue

        if not (
            is_side_effect_free(op)
            and is_speculatable(op)
            and can_be_hoisted(op, region)
        ):
            continue

        op.detach()
        builder.insert(op)

        # Since the op has been moved, we need to check its users within the
        # top-level of the loop body.

        for res in op.results:
            for use in res.uses:
                user = use.operation
                if user.parent_region() == region:
                    worklist.append(user)


def move_loop_invariant_code(loop: scf.ForOp):
    builder = Builder(InsertPoint.before(loop))
    _move_loop_invariant_code(loop.body, builder)


class LoopInvariantCodeMotion(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter) -> None:
        move_loop_invariant_code(op)


class LoopInvariantCodeMotionPass(ModulePass):
    """
    Moves operations without side effects out of loops, provided they do not depend on
    values defined in the loops.
    """

    name = "licm"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            LoopInvariantCodeMotion(),
            apply_recursively=False,
            walk_regions_first=True,
        ).rewrite_module(op)
