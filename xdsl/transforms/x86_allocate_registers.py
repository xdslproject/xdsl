from dataclasses import dataclass

from xdsl.backend.x86.register_allocation import X86RegisterAllocator
from xdsl.backend.x86.register_queue import X86RegisterQueue
from xdsl.context import Context
from xdsl.dialects import x86_func
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class X86AllocateRegisters(ModulePass):
    """
    Allocates unallocated registers in the module.
    """

    name = "x86-allocate-registers"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for inner_op in op.walk():
            if isinstance(inner_op, x86_func.FuncOp):
                register_queue = X86RegisterQueue.default()
                allocator = X86RegisterAllocator(register_queue)
                allocator.allocate_func(inner_op)
