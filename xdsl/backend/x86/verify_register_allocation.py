from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, x86_func
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class X86VerifyRegAlloc(ModulePass):
    name = "x86-verify-register-allocation"

    def _process_function(self, func: x86_func.FuncOp) -> None:
        pass

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for func in op.walk():
            if not isinstance(func, x86_func.FuncOp):
                continue

            if not func.body.blocks:
                continue

            self._process_function(func)
