from __future__ import annotations

from dataclasses import dataclass

from xdsl.backend.liveness import VerifyLivenessContext
from xdsl.context import Context
from xdsl.dialects import builtin, x86_func
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class X86RegallocVerifyLivenessPass(ModulePass):
    """
    Verify that the use of a register value as inout is its last use.
    """

    name = "x86-regalloc-verify-liveness"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for func in op.body.block.ops:
            if not isinstance(func, x86_func.FuncOp):
                continue
            VerifyLivenessContext(set()).process_region(func.body)
