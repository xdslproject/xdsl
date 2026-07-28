from dataclasses import dataclass

from xdsl.backend.x86.arch import X86Arch
from xdsl.backend.x86.legalization import LegalizationContext
from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import builtin, x86_func
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException


@dataclass(frozen=True)
class X86RegallocLegalizePass(ModulePass):
    """
    Legalize x86 code before register allocation by inserting copies when an
    inout use is not the last use of a value.
    """

    name = "x86-regalloc-legalize"
    arch: str | None = None

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        arch = X86Arch.arch_for_name(self.arch)
        for func in op.walk():
            if not isinstance(func, x86_func.FuncOp):
                continue
            if not func.body.blocks:
                # External declaration
                continue
            if len(func.body.blocks) != 1:
                raise PassFailedException(
                    "Cannot yet legalize func with multiple blocks."
                )
            legalize_ctx = LegalizationContext(
                set(),
                arch,
                Builder(InsertPoint.at_end(func.body.block)),
            )
            legalize_ctx.process_block(func.body.block)
