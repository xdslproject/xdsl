from typing import Any

from xdsl.dialects import riscv
from xdsl.interpreter import Interpreter, PythonValues, impl, register_impls
from xdsl.interpreters.ptr import RawPtr
from xdsl.interpreters.riscv import RiscvFunctions


@register_impls
class ToyAcceleratorInstructionFunctions(RiscvFunctions):
    def __init__(self):
        super().__init__(
            custom_instructions={
                "buffer.alloc": accelerator_buffer_alloc,
                "buffer.copy": accelerator_buffer_copy,
            },
        )

    @impl(riscv.EcallOp)
    def run_ecall(
        self,
        interpreter: Interpreter,
        op: riscv.EcallOp,
        args: tuple[Any, ...],
    ):
        # In Toy, ecall is always exit
        return ()


def accelerator_buffer_copy(
    interpreter: Interpreter, op: riscv.CustomAssemblyInstructionOp, args: PythonValues
) -> PythonValues:
    size, dest_buffer, source_buffer = args

    for i in range(size):
        dest_buffer[i] = source_buffer[i]

    return ()


def accelerator_buffer_alloc(
    interpreter: Interpreter, op: riscv.CustomAssemblyInstructionOp, args: PythonValues
) -> PythonValues:
    (size,) = args
    return (RawPtr.zeros(size * 8),)
