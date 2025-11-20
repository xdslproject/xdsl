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
        for op in reversed(func.body.ops):
            if isinstance(op, HasRegisterConstraints):
                ins, outs, inouts = op.get_register_constraints()
                ins = list(ins)
                outs = list(outs)
                for inout in inouts:
                    in_reg, out_reg = inout[0], inout[1]
                    if in_reg in alive:
                        raise DiagnosticException(
                            f"{in_reg.name_hint} should not be read after in/out usage"
                        )
                    ins.append(in_reg)
                    outs.append(out_reg)
                for o in outs:
                    if o in alive:
                        alive.remove(o)
                for i in ins:
                    alive.add(i)

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for func in op.walk():
            if not isinstance(func, x86_func.FuncOp):
                continue

            if not func.body.blocks:
                continue

            self._process_function(func)
