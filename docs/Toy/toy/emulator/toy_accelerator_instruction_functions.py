from typing import Any

from xdsl.dialects import riscv
from xdsl.interpreter import Interpreter, impl, register_impls
from xdsl.interpreters.riscv import RiscvFunctions


@register_impls
class ToyAcceleratorInstructionFunctions(RiscvFunctions):
    @impl(riscv.EcallOp)
    def run_ecall(
        self,
        interpreter: Interpreter,
        op: riscv.EcallOp,
        args: tuple[Any, ...],
    ):
        # In Toy, ecall is always exit
        return ()
