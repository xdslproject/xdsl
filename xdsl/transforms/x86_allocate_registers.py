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

    The set of allocatable registers depends on the target, which is read from
    the `x86.arch` module attribute. Without it the conservative VEX-only set is
    used, since handing out EVEX-only registers produces assembly that faults on
    hardware that does not have them.
    """

    name = "x86-allocate-registers"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        arch = X86Arch.from_module(op)
        for inner_op in op.walk():
            if isinstance(inner_op, x86_func.FuncOp):
                available_registers = X86RegisterStack.get_for_arch(arch)
                allocator = X86RegisterAllocator(available_registers)
                allocator.allocate_func(inner_op)
