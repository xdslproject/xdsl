from xdsl.backend.register_allocatable import RegisterAllocatableOperation
from xdsl.backend.register_allocator import BlockAllocator, live_ins_per_block
from xdsl.ir import Block, Region
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

    def allocate_region(self, body: Region) -> None:
        """
        Allocates values in given region to registers.
        """
        if len(body.blocks) != 1:
            raise NotImplementedError(
                f"Cannot register allocate region with {len(body.blocks)} blocks."
            )

        preallocated = RegisterAllocatableOperation.all_used_registers(body)
        excluded = RegisterAllocatableOperation.all_excluded_registers(body)

        for pa_reg in preallocated | excluded:
            self.available_registers.exclude_register(pa_reg)

        block = body.block

        self.live_ins_per_block = live_ins_per_block(block)
        assert not self.live_ins_per_block[block]

        self.allocate_block(block)
