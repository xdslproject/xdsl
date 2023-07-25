import array
from io import StringIO
from math import prod
from typing import Any, Sequence, cast

import wgpu
import wgpu.backends.rs
import wgpu.utils
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
    device = wgpu.utils.get_default_device()
    shader_modules: dict[gpu.ModuleOp] = {}

    def buffer_from_operand(self, interpreter: Interpreter, operand: SSAValue):
        if isa(operand.type, MemRefType[Attribute]):
            element_type = operand.type.element_type
            if isinstance(element_type, IndexType):
                shaped_array = interpreter.get_values((operand,))[0]
                assert isinstance(shaped_array, ShapedArray)
                values = tuple(
                    shaped_array.load(index) for index in shaped_array.indices()
                )
                view = memoryview(bytearray(len(values) * 4)).cast("I")
                buffer = self.device.create_buffer_with_data(
                    data=view, usage=wgpu.BufferUsage.STORAGE
                )
                return buffer

    def prepare_bindings(
        self, interpreter: Interpreter, kernel_operands: Sequence[SSAValue]
    ):
        layouts = []
        bindings = []
        for i, o in enumerate(kernel_operands):
            buffer = WGPUFunctions.buffer_from_operand(self, interpreter, o)
            layouts.append(
                {
                    "binding": i,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                }
            )
            bindings.append(
                {
                    "binding": i,
                    "resource": {"buffer": buffer, "offset": 0, "size": buffer.size},
                }
            )

        return layouts, bindings

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
        kernel_operands = op.kernelOperands

        func = SymbolTable.lookup_symbol(op, op.kernel)
        assert isinstance(func, gpu.FuncOp), func
        module = func.parent_op()
        assert isinstance(module, gpu.ModuleOp)
        interpreter.run_op(module, ())
        shader_module = self.shader_modules[module]

        layouts, bindings = WGPUFunctions.prepare_bindings(
            self, interpreter, kernel_operands
        )

        device = self.device
        # Put everything together
        bind_group_layout = device.create_bind_group_layout(entries=layouts)
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )
        bind_group = device.create_bind_group(
            layout=bind_group_layout, entries=bindings
        )

        # Create and run the pipeline
        compute_pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader_module, "entry_point": func.sym_name.data},
        )
        # assert compute_pipeline is None, compute_pipeline
        command_encoder = device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(compute_pipeline)
        compute_pass.set_bind_group(
            0, bind_group, [], 0, 999999
        )  # last 2 elements not used
        compute_pass.dispatch_workgroups(*gridSize)  # x y z
        compute_pass.end()
        device.queue.submit([command_encoder.finish()])
        return ()
        # for i, o in enumerate(kernel_operands):
        #     WGPUFunctions.process_outputs(self, interpreter, o, outputs[i + 1])
        # return ()
