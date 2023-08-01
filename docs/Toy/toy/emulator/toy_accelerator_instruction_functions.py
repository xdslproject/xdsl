from typing import Generic, TypeVar

from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter, PythonValues
from xdsl.interpreters.riscv import Buffer, RiscvFunctions
from xdsl.interpreters.shaped_array import ShapedArray

_T = TypeVar("_T")


class ShapedArrayBuffer(Generic[_T], ShapedArray[_T]):
    def __add__(self, offset: int) -> Buffer[_T]:
        return Buffer(self.data) + offset


class ToyAcceleratorInstructionFunctions(RiscvFunctions):
    def __init__(self, module_op: ModuleOp):
        super().__init__(
            module_op,
            custom_instructions={
                "tensor.print2d": accelerator_tensor_print2d,
                "tensor.transpose2d": accelerator_tensor_transpose2d,
                "buffer.alloc": accelerator_buffer_alloc,
                "buffer.copy": accelerator_buffer_copy,
                "buffer.add": accelerator_buffer_add,
                "buffer.mul": accelerator_buffer_mul,
                "print": print_,
            },
        )


def print_(
    interpreter: Interpreter, op: riscv.CustomAssemblyInstructionOp, args: PythonValues
) -> PythonValues:
    interpreter.print(f"{args[0]}")
    return ()


def accelerator_tensor_print2d(
    interpreter: Interpreter, op: riscv.CustomAssemblyInstructionOp, args: PythonValues
) -> PythonValues:
    buffer, rows, cols = args
    shaped_array = ShapedArray([float(value) for value in buffer.data], [rows, cols])
    interpreter.print(f"{shaped_array}")
    return ()


def accelerator_tensor_transpose2d(
    interpreter: Interpreter, op: riscv.CustomAssemblyInstructionOp, args: PythonValues
) -> PythonValues:
    dest_buffer, source_buffer, rows, cols = args

    source_shaped_array = ShapedArray(source_buffer.data, [rows, cols])
    dest_shaped_array = ShapedArray(dest_buffer.data, [cols, rows])

    for row in range(rows):
        for col in range(cols):
            value = source_shaped_array.load((row, col))
            dest_shaped_array.store((col, row), value)

    return ()


def accelerator_buffer_copy(
    interpreter: Interpreter, op: riscv.CustomAssemblyInstructionOp, args: PythonValues
) -> PythonValues:
    size, dest_buffer, source_buffer = args

    for i in range(size):
        dest_buffer[i] = source_buffer[i]

    return ()


def accelerator_buffer_add(
    interpreter: Interpreter, op: riscv.CustomAssemblyInstructionOp, args: PythonValues
) -> PythonValues:
    size, dest_buffer, source_buffer = args

    for i in range(size):
        dest_buffer[i] += source_buffer[i]

    return ()


def accelerator_buffer_mul(
    interpreter: Interpreter, op: riscv.CustomAssemblyInstructionOp, args: PythonValues
) -> PythonValues:
    size, dest_buffer, source_buffer = args

    for i in range(size):
        dest_buffer[i] *= source_buffer[i]

    return ()


def accelerator_buffer_alloc(
    interpreter: Interpreter, op: riscv.CustomAssemblyInstructionOp, args: PythonValues
) -> PythonValues:
    (size,) = args
    return (Buffer([0] * size),)
