from io import StringIO
from typing import Any, cast

from wgpu.utils import compute_with_buffers
from xdsl.dialects import arith, gpu
from xdsl.dialects.builtin import AnyFloatAttr, AnyIntegerAttr
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.interpreters.experimental.wgsl_printer import WGSLPrinter
from xdsl.utils.hints import isa


@register_impls
class WGPUFunctions(InterpreterFunctions):
    @impl(gpu.LaunchFuncOp)
    def run_launch_func(
        self, interpreter: Interpreter, op: gpu.LaunchFuncOp, args: tuple[Any, ...]
    ):
        if op.asyncToken is not None:
            raise NotImplementedError(
                "The WGPU interpreter does not handle asynchronous GPU regions at the moment."
            )

        gridSize = interpreter.get_values((op.gridSizeX, op.gridSizeY, op.gridSizeZ))
        blockSize = interpreter.get_values(
            (op.blockSizeX, op.blockSizeY, op.blockSizeZ)
        )
        dispatch_count = tuple(g * b for g, b in zip(gridSize, blockSize))
        kernel_operands = interpreter.get_values(op.kernelOperands)

        func = interpreter.get_op_for_symbol(op.kernel.string_value().split(".")[-1])
        printer = WGSLPrinter()
        wgsl_source = StringIO("")
        printer.print(func, wgsl_source)
        print(f"Running {op.kernel.name} with dispatch count {dispatch_count}. Source:")
        print(wgsl_source.getvalue())
        operands_dict = {}
        for i, o in enumerate(kernel_operands):
            operands_dict[i] = o
        compute_with_buffers(
            operands_dict, {1: (1, "i")}, wgsl_source.getvalue(), dispatch_count
        )
        return (0,)
