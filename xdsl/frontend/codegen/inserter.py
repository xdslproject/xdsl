from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from xdsl.dialects.builtin import ModuleOp
from xdsl.frontend.codegen.exception import CodegenInternalException
from xdsl.ir import Block, Operation, Region, SSAValue


@dataclass
class OpInserter:
    """
    Class responsible for inserting operations at the right place (i.e. region,
    or block).
    """

    container: List[Operation] = field(default_factory=list)
    """
    Container for top-level operations in the current frontend program. The
    motivation for this is that we would like to have something like:

    with CodeContext(p):
        a: i32 = 23
        ...

    without requiring the top-level operation to be a module with functions.
    """

    stack: List[SSAValue] = field(default_factory=list)
    """
    Stack to hold the intermediate results of operations. For each new operation,
    its operands are popped from this stack.
    """

    ip: Block | None = field(default=None)
    """Insertion point, i.e. pointer to the block to which we append operations."""

    def get_operand(self) -> SSAValue:
        """Pops the last value from the operand stack."""
        if len(self.stack) == 0:
            raise CodegenInternalException("Trying to get an operand from empty stack.")
        return self.stack.pop()

    def insert_op(self, op: Operation) -> None:
        """Inserts a new operation."""

        # First, check if insertion point is set. If not, it means that this operation
        # is a top-level operation. Therefore, append it to the container.
        if self.ip is None:
            self.container.append(op)

            # Additionally, if this operation has a nested region/block, insert any future
            # operations there by default by setting the insertion point.
            if len(op.regions) != 0 and len(op.regions[-1].blocks) != 0:
                self.ip = op.regions[-1].blocks[-1]
        else:
            # This is not a top-level operation, so simply append to the end of the block.
            self.ip.add_op(op)

        # Finally, we push the result of the operation on the stack so that subsequent
        # operations can use it as operand.
        for result in op.results:
            self.stack.append(result)

    def set_insertion_point_from_op(self, op: Operation | None) -> None:
        """
        Reset insertion point to the last block in the last region of the operation.
        """

        # Special case: if operation is none, it means it is a top-level operation
        # and therefore insertion point should become None.
        if op is None:
            self.ip = None
            return

        # Otherwise, get the last region and the last block and set the insertion point
        # to it.
        if len(op.regions) == 0:
            raise CodegenInternalException("Trying to set insertion point for operation {} with no regions.", [op])
        if len(op.regions[-1].blocks) == 0:
            raise CodegenInternalException("Trying to set insertion point for operation {} with no blocks in the region.", [op])
        self.ip = op.regions[-1].blocks[-1]

    def set_insertion_point_from_region(self, region: Region) -> None:
        """Reset insertion point to the last block in this region."""
        if len(region.blocks) == 0:
            raise CodegenInternalException("Trying to set insertion point for the region with no blocks.")
        self.ip = region.blocks[-1]

    def set_insertion_point_from_block(self, block: Block) -> None:
        """Reset insertion point to this block."""
        self.ip = block
