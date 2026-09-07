from dataclasses import dataclass

from xdsl.backend.x86.arch import X86Arch
from xdsl.backend.x86.register_allocation import X86RegisterAllocator
from xdsl.backend.x86.register_stack import X86RegisterStack
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

    arch: str | None = None
    """
    The target architecture, which determines the allocatable register set. Without
    one, only the registers every x86 target can encode are handed out.
    """

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        arch = X86Arch.arch_for_name(self.arch)
        for inner_op in op.walk():
            if isinstance(inner_op, x86_func.FuncOp):
                available_registers = X86RegisterStack.get_for_arch(arch)
                allocator = X86RegisterAllocator(available_registers)
                allocator.allocate_func(inner_op)
