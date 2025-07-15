from xdsl.backend.register_allocatable import RegisterAllocatableOperation
from xdsl.backend.register_allocator import BlockAllocator
from xdsl.ir import Block
from xdsl.utils.exceptions import DiagnosticException


class BlockNaiveAllocator(BlockAllocator):
    """
    It traverses the use-def SSA chain backwards (i.e., from uses to defs) and:
      1. allocates registers for operands
      2. frees registers for results (since that will be the last time they appear when
      going backwards)
    It currently operates on blocks.

    This is a simplified version of the standard bottom-up local register allocation
    algorithm.

    A relevant reference in "Engineering a Compiler, Edition 3" ISBN: 9780128154120.

    ```
    for op in block.walk_reverse():
    for operand in op.operands:
        if operand is not allocated:
            allocate(operand)

    for result in op.results:
    if result is not allocated:
        allocate(result)
        free_before_next_instruction.append(result)
    else:
        free(result)
    ```
    """

    def allocate_block(self, block: Block):
        for op in reversed(block.ops):
            if isinstance(op, RegisterAllocatableOperation):
                try:
                    op.allocate_registers(self)
                except DiagnosticException as e:
                    op.emit_error("Error allocating op", e)
