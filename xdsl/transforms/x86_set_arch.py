"""
Record the x86 target on the module.

Passes downstream read it from there rather than each taking their own `arch`
option, in the same spirit as an LLVM module carrying its target triple.
"""

from dataclasses import dataclass

from xdsl.backend.x86.arch import X86Arch
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class X86SetArch(ModulePass):
    """
    Sets the `x86.arch` attribute on the module.
    """

    name = "x86-set-arch"

    arch: str

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        X86Arch.arch_for_name(self.arch).set_on_module(op)
