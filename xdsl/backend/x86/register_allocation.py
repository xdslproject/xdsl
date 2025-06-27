from xdsl.backend.block_naive_allocator import BlockNaiveAllocator
from xdsl.backend.register_allocatable import RegisterAllocatableOperation
from xdsl.backend.register_allocator import live_ins_per_block
from xdsl.backend.register_stack import RegisterStack
from xdsl.dialects import x86_func
from xdsl.dialects.x86 import register


class X86RegisterAllocator(BlockNaiveAllocator):
    def __init__(self, available_registers: RegisterStack) -> None:
        super().__init__(available_registers, register.X86RegisterType)

    def allocate_func(self, func: x86_func.FuncOp) -> None:
        """
        Allocates values in function passed in to registers.
        The whole function must have been lowered to the relevant x86 dialects
        and it must contain no unrealized casts.
        """
        if not func.body.blocks:
            # External function declaration
            return

        if len(func.body.blocks) != 1:
            raise NotImplementedError(
                f"Cannot register allocate func with {len(func.body.blocks)} blocks."
            )

        preallocated = {
            reg
            for reg in RegisterAllocatableOperation.iter_all_used_registers(func.body)
            if isinstance(reg, register.X86RegisterType)
        }

        for pa_reg in preallocated:
            self.available_registers.exclude_register(pa_reg)

        block = func.body.block

        self.live_ins_per_block = live_ins_per_block(block)
        assert not self.live_ins_per_block[block]

        self.allocate_block(block)
