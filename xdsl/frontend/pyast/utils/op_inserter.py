from dataclasses import dataclass, field

from xdsl.frontend.pyast.utils.exceptions import FrontendProgramException
from xdsl.ir import Block, Operation, Region, SSAValue


@dataclass
class OpInserter:
    """
    Class responsible for inserting operations at the right place in the
    generated IR.
    """

    insertion_point: Block
    """
    Insertion point, i.e. the pointer to the block to which the operations are
    appended.
    """

    stack: list[SSAValue] = field(default_factory=list[SSAValue])
    """
    Stack to hold the intermediate results of operations. For each new
    operation, its operands will be popped from the stack.
    """

    def get_operand(self) -> SSAValue:
        """
        Pops the last value from the operand stack and returns it.
        """
        if len(self.stack) == 0:
            raise FrontendProgramException(
                "Trying to get an operand from an empty stack."
            )
        return self.stack.pop()

    def insert_op(self, op: Operation) -> None:
        """Inserts a new operation and places its results on the stack."""
        self.insertion_point.add_op(op)
        for result in op.results:
            self.stack.append(result)

    def set_insertion_point_from_op(self, op: Operation) -> None:
        """
        Sets the insertion point to the last block in the last region of the
        operation.
        """
        if not op.regions:
            raise FrontendProgramException(
                f"Trying to set the insertion point for operation '{op.name}' with no regions."
            )
        if (last_block := op.regions[-1].blocks.last) is None:
            raise FrontendProgramException(
                f"Trying to set the insertion point for operation '{op.name}' with no blocks in its last region."
            )
        self.insertion_point = last_block

    def set_insertion_point_from_region(self, region: Region) -> None:
        """Sets the insertion point to the last block in this region."""
        if (last_block := region.blocks.last) is None:
            raise FrontendProgramException(
                "Trying to set the insertion point from the region without blocks."
            )
        self.insertion_point = last_block

    def set_insertion_point_from_block(self, block: Block) -> None:
        """Sets the insertion point to this block."""
        self.insertion_point = block
