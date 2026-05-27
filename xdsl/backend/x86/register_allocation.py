from xdsl.backend.block_naive_allocator import BlockNaiveAllocator
from xdsl.backend.register_stack import RegisterStack
from xdsl.dialects.x86 import registers


class X86RegisterAllocator(BlockNaiveAllocator):
    def __init__(self, available_registers: RegisterStack) -> None:
        super().__init__(available_registers, registers.X86RegisterType)
