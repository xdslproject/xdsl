from dataclasses import dataclass

from xdsl.backend.register_allocatable import RegisterAllocatableContentsOperation
from xdsl.backend.x86.register_stack import X86RegisterStack
from xdsl.context import Context
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
            if isinstance(inner_op, RegisterAllocatableContentsOperation):
                available_registers = X86RegisterStack.get()
                inner_op.allocate_registers(available_registers)
