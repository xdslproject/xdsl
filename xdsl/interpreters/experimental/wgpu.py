import array
from io import StringIO
from math import prod
from typing import Any, cast

import wgpu
import wgpu.backends.rs
from xdsl.dialects import arith, gpu
from xdsl.dialects.builtin import AnyFloatAttr, AnyIntegerAttr, IndexType
from xdsl.dialects.memref import MemRefType
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.interpreters.experimental.wgsl_printer import WGSLPrinter
from xdsl.interpreters.memref import MemrefValue
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.ir.core import Attribute, SSAValue
from xdsl.traits import SymbolTable
from xdsl.utils.hints import isa


@register_impls
class WGPUFunctions(InterpreterFunctions):
    def __init__(self):
        self.device = wgpu.utils.get_default_device()
        self.shader_modules: dict[gpu.ModuleOp]

    def prepare_operand(self, interpreter: Interpreter, operand: SSAValue):
        if isa(operand.type, MemRefType[Attribute]):
            element_type = operand.type.element_type
            if isinstance(element_type, IndexType):
                shaped_array = interpreter.get_values((operand,))[0]
                assert isinstance(shaped_array, ShapedArray)
                values = (shaped_array.load(index) for index in shaped_array.indices())
                return (
                    array.array("I", (v if isinstance(v, int) else 0 for v in values)),
                    (*shaped_array.shape, "I"),
                )

    def process_outputs(self, interpreter: Interpreter, operand: SSAValue, output: Any):
        if isa(operand.type, MemRefType[Attribute]):
            element_type = operand.type.element_type
            if isinstance(element_type, IndexType):
                assert isinstance(output, memoryview)
                shaped_array = interpreter.get_values((operand,))[0]
                assert isinstance(shaped_array, ShapedArray)
                for index in shaped_array.indices():
                    shaped_array.store(index, output[index])
                    pass

    @impl(gpu.ModuleOp)
    def compile_module(
        self, interpreter: Interpreter, op: gpu.ModuleOp, args: tuple[()]
    ):
        if op not in self.shader_modules:
            printer = WGSLPrinter()
            wgsl_source = StringIO("")
            printer.print(op, wgsl_source)
            self.shader_modules[op] = self.device.create_shader_module(
                code=wgsl_source.getvalue()
            )
        return ()

    @impl(gpu.LaunchFuncOp)
    def run_launch_func(
        self, interpreter: Interpreter, op: gpu.LaunchFuncOp, args: tuple[Any, ...]
    ):
        if op.asyncToken is not None or len(op.asyncDependencies) != 0:
            raise NotImplementedError(
                "The WGPU interpreter does not handle asynchronous GPU regions at the moment."
            )

        gridSize = interpreter.get_values((op.gridSizeX, op.gridSizeY, op.gridSizeZ))
        blockSize = interpreter.get_values(
            (op.blockSizeX, op.blockSizeY, op.blockSizeZ)
        )
        dispatch_count = tuple(g * b for g, b in zip(gridSize, blockSize))
        kernel_operands = op.kernelOperands

        func = SymbolTable.lookup_symbol(op, op.kernel)
        assert isinstance(func, gpu.FuncOp)
        module = func.parent_op()
        assert isinstance(module, gpu.ModuleOp)
        interpreter.run_op(module, ())
        shader_module = self.shader_modules[module]

        operands_dict = {}
        outputs_dict = {}
        for i, o in enumerate(kernel_operands):
            operand, output = WGPUFunctions.prepare_operand(self, interpreter, o)
            operands_dict[i + 1] = operand
            outputs_dict[i + 1] = output
        outputs = compute_with_buffers(
            operands_dict, outputs_dict, wgsl_source.getvalue(), dispatch_count
        )
        for i, o in enumerate(kernel_operands):
            WGPUFunctions.process_outputs(self, interpreter, o, outputs[i + 1])
        return ()
