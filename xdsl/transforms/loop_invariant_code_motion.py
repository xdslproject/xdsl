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

from xdsl.ir import Operation, Region
from xdsl.traits import IsTerminator


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
            assert operand_owner is not None
            if op.is_ancestor(operand_owner):
                continue
            if target_region.is_ancestor(operand_owner):
                return False
    return True
