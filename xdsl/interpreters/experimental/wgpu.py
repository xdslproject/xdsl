import array
from io import StringIO
from math import prod
from typing import Any, Sequence, cast

import wgpu
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
    device = cast(wgpu.GPUDevice, wgpu.utils.get_default_device())
    shader_modules: dict[gpu.FuncOp, wgpu.GPUShaderModule] = {}

    def buffer_from_operand(self, interpreter: Interpreter, operand: SSAValue):
        if isa(operand.type, MemRefType[Attribute]):
            element_type = operand.type.element_type
            if isinstance(element_type, IndexType):
                shaped_array = interpreter.get_values((operand,))[0]
                assert isinstance(shaped_array, ShapedArray)
                values = tuple(
                    shaped_array.load(index) for index in shaped_array.indices()
                )
                view = memoryview(bytearray(len(values) * 4)).cast(
                    "I", shaped_array.shape
                )
                for idx in shaped_array.indices():
                    v = shaped_array.load(idx)
                    view[idx] = v if isinstance(v, int) else 0
                buffer = cast(
                    wgpu.GPUBuffer,
                    self.device.create_buffer_with_data(
                        data=view,
                        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                    ),
                )
                return buffer
        raise NotImplementedError(f"{operand.type} not yet mapped to WGPU.")

    def prepare_bindings(
        self, interpreter: Interpreter, kernel_operands: Sequence[SSAValue]
    ):
        layouts: list[dict[str, Any]] = []
        bindings: list[dict[str, Any]] = []
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

    def process_bindings(
        self,
        interpreter: Interpreter,
        launch: gpu.LaunchFuncOp,
        bindings: list[dict[str, Any]],
    ):
        for i, binding in enumerate(bindings):
            operand = launch.kernelOperands[i]
            gpu_buffer = binding["resource"]["buffer"]
            buffer = cast(memoryview, self.device.queue.read_buffer(gpu_buffer))
            if isa(operand.type, MemRefType[Attribute]):
                element_type = operand.type.element_type
                if isinstance(element_type, IndexType):
                    buffer = buffer.cast(
                        "I", [i.value.data for i in operand.type.shape]
                    )
                    # print(buffer.tolist())
                    value = interpreter.get_values((operand,))[0]
                    assert isinstance(value, ShapedArray), value
                    for index in value.indices():
                        value.store(index, buffer.__getitem__(index))
                    print(value.data)

    def compile_func(self, interpreter: Interpreter, op: gpu.FuncOp):
        if op not in self.shader_modules:
            wgsl_printer = WGSLPrinter()
            wgsl_source = StringIO("")
            wgsl_printer.print(op, wgsl_source)
            print(f"Compiling:\n{wgsl_source.getvalue()}")
            self.shader_modules[op] = self.device.create_shader_module(
                code=wgsl_source.getvalue()
            )

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
        assert isinstance(func, gpu.FuncOp)
        WGPUFunctions.compile_func(self, interpreter, func)
        shader_module = self.shader_modules[func]

        # Compute the dispatch number
        dispatch = list(gridSize)
        if not func.known_block_size:
            for i in range(len(dispatch)):
                dispatch[i] = dispatch[i] * blockSize[i]

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

        command_encoder = device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(compute_pipeline)
        compute_pass.set_bind_group(
            0, bind_group, [], 0, 999999
        )  # last 2 elements not used
        compute_pass.dispatch_workgroups(*dispatch)  # x y z
        compute_pass.end()
        device.queue.submit([command_encoder.finish()])

        WGPUFunctions.process_bindings(self, interpreter, op, bindings)

        return ()
