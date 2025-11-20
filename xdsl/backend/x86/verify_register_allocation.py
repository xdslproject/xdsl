from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, x86_func
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import DiagnosticException


@dataclass(frozen=True)
class X86VerifyRegAlloc(ModulePass):
    name = "x86-verify-register-allocation"

    def _process_function(self, func: x86_func.FuncOp) -> None:
        alive: set[SSAValue] = set()
        for op in reversed(func.body.ops):
            for r in op.results:
                if r in alive:
                    alive.remove(r)
            for o in op.operands:
                if o.type in op.result_types and o in alive:
                    raise DiagnosticException(
                        f"{o.name_hint} should not be read after in/out usage"
                    )
                alive.add(o)

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for func in op.walk():
            if not isinstance(func, x86_func.FuncOp):
                continue

            if not func.body.blocks:
                continue

            self._process_function(func)
