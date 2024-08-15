from collections.abc import Sequence
from io import StringIO
from typing import Any, cast

import wgpu  # pyright: ignore[reportMissingTypeStubs]
import wgpu.utils  # pyright: ignore[reportMissingTypeStubs]

from xdsl.dialects import gpu
from xdsl.dialects.builtin import IndexType
from xdsl.dialects.memref import MemRefType
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.interpreters.experimental.wgsl_printer import WGSLPrinter
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.ir import Attribute, SSAValue
from xdsl.traits import SymbolTable
from xdsl.utils.hints import isa


@register_impls
class WGPUFunctions(InterpreterFunctions):
    device = cast(wgpu.GPUDevice, wgpu.utils.get_default_device())
    shader_modules: dict[gpu.FuncOp, wgpu.GPUShaderModule] = {}

    def buffer_from_operand(self, interpreter: Interpreter, operand: SSAValue):
        """
        Prepare a GPUBuffer from an SSA operand.

        memrefs are excpected to be GPUBuffers at this point.

        Still a helper function, because boilerplating will need to happen to forward
        e.g. scalar parameters!
        """
        if isa(operand.type, MemRefType[Attribute]):
            value = interpreter.get_values((operand,))[0]
            if not isinstance(value, wgpu.GPUBuffer):
                raise ValueError(
                    "gpu.launch_func memref operand expected to be GPU-allocated"
                )
            return value
        raise NotImplementedError(f"{operand.type} not yet mapped to WGPU.")

    def prepare_bindings(
        self, interpreter: Interpreter, kernel_operands: Sequence[SSAValue]
    ):
        """
        Boilerplate preparation for arguments bindings.
        """
        layouts: list[dict[str, Any]] = []
        bindings: list[dict[str, Any]] = []
        for i, o in enumerate(kernel_operands):
            buffer = WGPUFunctions.buffer_from_operand(self, interpreter, o)

            layouts.append(
                {
                    "binding": i,
                    "visibility": wgpu.ShaderStage.COMPUTE,  # pyright: ignore
                    "buffer": {
                        "type": wgpu.BufferBindingType.storage  # pyright: ignore
                    },
                }
            )
            bindings.append(
                {
                    "binding": i,
                    "resource": {
                        "buffer": buffer,
                        "offset": 0,
                        "size": buffer.size,  # pyright: ignore
                    },
                }
            )

        return layouts, bindings

    def compile_func(self, op: gpu.FuncOp):
        """
        Compile a gpu.func if not already done.
        """
        if op not in self.shader_modules:
            wgsl_printer = WGSLPrinter()
            wgsl_source = StringIO("")
            wgsl_printer.print(op, wgsl_source)
            self.shader_modules[op] = cast(
                wgpu.GPUShaderModule,
                self.device.create_shader_module(  # pyright: ignore
                    code=wgsl_source.getvalue()
                ),  # pyright: ignore
            )

    @impl(gpu.AllocOp)
    def run_alloc(
        self, interpreter: Interpreter, op: gpu.AllocOp, args: tuple[Any, ...]
    ):
        """
        Allocate a GPUBuffer according to a gpu.alloc operation, return it as the memref
        value.
        """
        if args or op.asyncToken:
            raise NotImplementedError(
                "Only synchronous, known-sized gpu.alloc implemented yet."
            )
        memref_type = cast(MemRefType[Attribute], op.result.type)
        match memref_type.element_type:
            case IndexType():
                element_size = 4
            case _:
                raise NotImplementedError(
                    f"The element type {memref_type.element_type} for gpu.alloc is not implemented yet."
                )
        buffer = cast(
            wgpu.GPUBuffer,
            self.device.create_buffer(  # pyright: ignore
                size=memref_type.element_count() * element_size,
                usage=wgpu.BufferUsage.STORAGE  # pyright: ignore
                | wgpu.BufferUsage.COPY_SRC,  # pyright: ignore
            ),
        )
        return (buffer,)

    @impl(gpu.MemcpyOp)
    def run_memcpy(
        self, interpreter: Interpreter, op: gpu.MemcpyOp, args: tuple[Any, ...]
    ) -> tuple[()]:
        """
        Copy buffers according to the gpu.memcpy operation.

        Only Device to Host copy is implemented here, to keep the first draft bearable.
        """
        src, dst = interpreter.get_values((op.src, op.dst))
        if not (isinstance(src, wgpu.GPUBuffer) and isinstance(dst, ShapedArray)):
            raise NotImplementedError(
                f"Only device to host copy is implemented for now. got {src} to {dst}"
            )

        # Get device/source view
        memview = cast(
            memoryview,
            self.device.queue.read_buffer(src),  # pyright: ignore
        )
        dst_type = cast(MemRefType[Attribute], op.dst.type)
        match dst_type.element_type:
            case IndexType():
                format = "I"
            case _:
                raise NotImplementedError(
                    f"copy for element type {dst_type.element_type} not yet implemented."
                )
        memview = memview.cast(format, dst_type.get_shape())  # pyright: ignore
        for index in dst.indices():
            dst.store(index, memview.__getitem__(index))  # pyright: ignore
        return ()

    @impl(gpu.LaunchFuncOp)
    def run_launch_func(
        self, interpreter: Interpreter, op: gpu.LaunchFuncOp, args: tuple[Any, ...]
    ):
        """
        Launch a GPU kernel through the WebGPU API.
        """
        if op.asyncToken is not None or op.asyncDependencies:
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
        WGPUFunctions.compile_func(self, func)
        shader_module = self.shader_modules[func]

        # Compute the dispatch number
        # If the func has a known block size, it's reflected in the compiled module
        # Otherwise, it defaults to (1,1,1) currently and we have to take this
        # into account
        if func.known_block_size:
            dispatch = gridSize
        else:
            dispatch = [a * b for a, b in zip(gridSize, blockSize)]

        layouts, bindings = WGPUFunctions.prepare_bindings(
            self, interpreter, kernel_operands
        )

        # All the boilerplate
        device = self.device
        # Put bindings together
        bind_group_layout = device.create_bind_group_layout(  # pyright: ignore
            entries=layouts
        )
        pipeline_layout = device.create_pipeline_layout(  # pyright: ignore
            bind_group_layouts=[bind_group_layout]
        )
        bind_group = device.create_bind_group(  # pyright: ignore
            layout=bind_group_layout,  # pyright: ignore
            entries=bindings,
        )

        # Create and run the pipeline
        compute_pipeline = device.create_compute_pipeline(  # pyright: ignore
            layout=pipeline_layout,  # pyright: ignore
            compute={"module": shader_module, "entry_point": func.sym_name.data},
        )

        command_encoder = device.create_command_encoder()  # pyright: ignore
        compute_pass = command_encoder.begin_compute_pass()  # pyright: ignore
        compute_pass.set_pipeline(compute_pipeline)  # pyright: ignore
        compute_pass.set_bind_group(  # pyright: ignore
            0, bind_group, [], 0, 0
        )  # last 2 elements not used
        compute_pass.dispatch_workgroups(*dispatch)  # x y z # pyright: ignore
        compute_pass.end()  # pyright: ignore
        device.queue.submit([command_encoder.finish()])  # pyright: ignore

        # gpu.launch_func has no return
        return ()
