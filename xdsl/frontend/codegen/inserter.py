from dataclasses import dataclass, field
from typing import List
from xdsl.dialects import builtin, func, symref
from xdsl.frontend.codegen.exception import CodegenException
from xdsl.ir import Block, Operation, Region, SSAValue


@dataclass
class DeclareInsertionPoint:
    """Represents insertion point for symref declarations."""

    block: Block | None = field(default=None)
    """Block where to insert declaration."""

    index: int = field(default=0)
    """Index at which insert the declaration in the block."""

    def insert(self, op: symref.Declare):
        """
        Inserts a symref declaration to the top-level block. What the top-level
        block would be? It would be the "closest" function or module op.
        """
        assert self.block is not None
        self.block.insert_op(op, self.index)
        self.index += 1


@dataclass
class OpInserter:
    """
    Class responsible for inserting operations at the right place (i.e. region,
    or block).
    """

    op_container: List[Operation] = field(default_factory=list)
    """
    Container for top-level operations in the current frontend program. The
    motivation for this is that we would acc to have something like:

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
    """Block to which we append operations."""

    declare_ip: DeclareInsertionPoint = field(default_factory=DeclareInsertionPoint)
    """Index for the next semref declaration."""

    def get_operand(self) -> SSAValue:
        """Pops the last value from the operand stack."""
        if len(self.stack) == 0:
            raise CodegenException("trying to get an operand from empty stack")

        return self.stack.pop()

    def insert_op(self, op: Operation):
        """Inserts a new operation."""

        # First, we check if this operation resets the insertion point for
        # symref declarations.
        if isinstance(op, builtin.ModuleOp) or isinstance(op, func.FuncOp):
            self.declare_ip = DeclareInsertionPoint(op.body.blocks[0], 0)

        # Then, check if insertion point is set. If not, it means that this operation
        # is a top-level operation. Therefore, append it to the container.
        if self.ip is None:
            self.op_container.append(op)

            # Additionally, if this operation has a region/block, insert any future
            # operations there by default.
            if len(op.regions) != 0 and len(op.regions[-1].blocks) != 0:
                self.ip = op.regions[-1].blocks[-1]
        else:
            # Otherwise, also check if operation is a symref declaration. In that case
            # we want to put all declaration in the enclosing module or function.
            if isinstance(op, symref.Declare):
                self.declare_ip.insert(op)
            else:
                self.ip.add_op(op)

        # Last, we push the result of the operation on the stack so that subsequent
        # operations can use it as operand.
        if len(op.results) > 1:
            raise CodegenException(f"expected {op} to return a single result, but \
                                     got {len(op.results)}")
        for result in op.results:
            self.stack.append(result)

    def set_insertion_point_from_op(self, op: Operation | None):
        """
        Reset insertion point to the last block in the last region of the
        operation.
        """

        # Special case: if operation is none, it means it is a top-level operation
        # and therefore insertion point should become none.
        if op is None:
            self.ip = None
            return

        # Otherwise, get the last region and the last block and set insertion point
        # to it.
        if len(op.regions) == 0:
            raise CodegenException(f"cannot set insertion point because {op} does \
                                     not have regions")
        if len(op.regions[-1].blocks) == 0:
            raise CodegenException(f"cannot set insertion point because  {op} does \
                                     not have blocks")
        self.ip = op.regions[-1].blocks[-1]

    def set_insertion_point_from_region(self, region: Region):
        """Reset insertion point to the last block in this region."""
        if len(region.blocks) == 0:
            raise CodegenException(f"{region} does not have blocks")
        self.ip = region.blocks[-1]

    def set_insertion_point_from_block(self, block: Block):
        """Reset insertion point to this block."""
        self.ip = block

    def set_declare_insertion_point(self, point: DeclareInsertionPoint):
        self.declare_ip = point
