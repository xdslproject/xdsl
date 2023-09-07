from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter, PythonValues
from xdsl.interpreters.riscv import RawPtr, RiscvFunctions
from xdsl.interpreters.shaped_array import ShapedArray


class ToyAcceleratorInstructionFunctions(RiscvFunctions):
    def __init__(self, module_op: ModuleOp):
        super().__init__(
            module_op,
            custom_instructions={
                "tensor.print1d": accelerator_tensor_print1d,
                "tensor.print2d": accelerator_tensor_print2d,
                "print": print_,
            },
        )


def print_(
    interpreter: Interpreter, op: riscv.CustomAssemblyInstructionOp, args: PythonValues
) -> PythonValues:
    interpreter.print(f"{args[0]}")
    return ()


def accelerator_tensor_print1d(
    interpreter: Interpreter, op: riscv.CustomAssemblyInstructionOp, args: PythonValues
) -> PythonValues:
    assert len(args) == 2
    ptr: RawPtr = args[0]
    els: int = args[1]

    shaped_array = ShapedArray(ptr.float32.get_list(els), [els])
    interpreter.print(f"{shaped_array}")
    return ()


def accelerator_tensor_print2d(
    interpreter: Interpreter, op: riscv.CustomAssemblyInstructionOp, args: PythonValues
) -> PythonValues:
    assert len(args) == 3
    ptr: RawPtr = args[0]
    rows: int = args[1]
    cols: int = args[2]
    shaped_array = ShapedArray(ptr.float32.get_list(rows * cols), [rows, cols])
    interpreter.print(f"{shaped_array}")
    return ()
