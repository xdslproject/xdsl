from dataclasses import dataclass

from xdsl.backend.register_allocatable import HasRegisterConstraints
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
        for op in func.body.walk(reverse=True):
            alive.difference_update(op.results)
            if isinstance(op, HasRegisterConstraints):
                _, _, inouts = op.get_register_constraints()
                for in_reg, _ in inouts:
                    if in_reg in alive:
                        raise DiagnosticException(
                            f"{in_reg.name_hint} should not be read after in/out usage"
                        )
            alive = alive.union(op.operands)

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for func in op.walk():
            if not isinstance(func, x86_func.FuncOp):
                continue

            if not func.body.blocks:
                continue

            self._process_function(func)
