"""
Example compute shader that does ... nothing but copy a value from one
buffer into another.
"""
from io import StringIO
from sys import stdout

import wgpu
import wgpu.backends.rs  # Select backend
from wgpu.utils import compute_with_buffers  # Convenience function

# from xdsl.dialects.builtin import FunctionType, SymbolRefAttr
# from xdsl.dialects.func import FuncOp  # Convenience function
# from xdsl.dialects import gpu, memref, builtin
# from xdsl.builder import Builder
# from xdsl.ir import BlockArgument
# from xdsl.printer import Printer
# from xdsl.interpreter import Interpreter


# This defines the function's body
# @Builder.implicit_region(
#     [
#         # It takes two memref<?xi32> (i.e. two dynamic arrays of i32)
#         memref.MemRefType.from_element_type_and_shape(builtin.i32, shape=[-1]),
#         memref.MemRefType.from_element_type_and_shape(builtin.i32, shape=[-1]),
#     ]
# )
# def function_body(args: tuple[BlockArgument, ...]):
#     # Find out the thread id in dim x
#     # to mimic WGPU "let i: u32 = index.x;"
#     thread_id_x = gpu.GlobalIdOp.get(gpu.DimensionAttr.from_dimension("x"))
#     # That's just setting
#     # the printed name to "i" for convenience
#     thread_id_x.result.name_hint = "i"
#     # Take the value from
#     load = memref.Load.get(args[0], [thread_id_x.result])
#     load.res.name_hint = "val"
#     memref.Store.get(load.res, args[1], [thread_id_x.result])
#
#
# @Builder.implicit_region
# def module_region():
#     FuncOp(
#         "main",
#         FunctionType.from_lists(
#             [
#                 memref.MemRefType.from_element_type_and_shape(builtin.i32, shape=[-1]),
#                 memref.MemRefType.from_element_type_and_shape(builtin.i32, shape=[-1]),
#             ],
#             [],
#         ),
#         function_body,
#     )


# shader_ir = gpu.ModuleOp.get(SymbolRefAttr("main"), module_region)
# out_string = StringIO("")
# wgpu_printer = WGPUFunctions()
# wgpu_printer.print(shader_ir, out_string)
# print(out_string.getvalue())

# %% Shader and data

shader_source = """

@group(0) @binding(0)
var<storage,read> data1: array<i32>;

@group(0) @binding(0)
var<storage,read_write> data2: array<i32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {
    let i: u32 = index.x;
    data2[i] = data1[i];
}
"""

new = """
    
        @group(0) @binding(0)
    var<storage,read> varg0: u32;

    @group(0) @binding(1)
    var<storage,read> varg1: u32;

    @group(0) @binding(2)
    var<storage,read> varg2: u32;

    @group(0) @binding(3)
    var<storage,read> varg3: array<f32>;

    @group(0) @binding(4)
    var<storage,read> varg4: u32;

    @group(0) @binding(5)
    var<storage,read> varg5: f32;

    @group(0) @binding(6)
    var<storage,read> varg6: f32;

    @group(0) @binding(7)
    var<storage,read> varg7: f32;

    @group(0) @binding(8)
    var<storage,read> varg8: f32;

    @group(0) @binding(9)
    var<storage,read> varg9: f32;

    @group(0) @binding(10)
    var<storage,read> varg10: f32;

    @group(0) @binding(11)
    var<storage,read_write> varg11: array<f32>;


    @compute
    @workgroup_size(128,1,1)
    fn main(@builtin(global_invocation_id) global_invocation_id : vec3<u32>,
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id : vec3<u32>,
    @builtin(num_workgroups) num_workgroups : vec3<u32>) {

        let v0 : u32 = 2u;
        let v1: u32 = workgroup_id.x;
        let v2: u32 = workgroup_id.y;
        let v3: u32 = local_invocation_id.x;
        let v4: u32 = local_invocation_id.y;
        let v5 = v1 * varg0;
        let v6 = v5 + varg1;
        let v7 = v2 * varg2;
        let v8 = v7 + varg1;
        let v9 = v3 * varg2;
        let v10 = v9 + varg1;
        let v11 = v4 * varg2;
        let v12 = v11 + varg1;
        let v13 = v10 + v6;
        let v14 = v12 + v8;
        let v15 = v14 + v0;
        let v16 = v13 + v0;
        let v17 = varg3[260u * v15 + 1u * v16];
        let v18 = v14 + varg4;
        let v19 = v18 + v0;
        let v20 = varg3[260u * v19 + 1u * v16];
        let v21 = v14 + varg2;
        let v22 = v21 + v0;
        let v23 = varg3[260u * v22 + 1u * v16];
        let v24 = v13 + varg4;
        let v25 = v24 + v0;
        let v26 = varg3[260u * v15 + 1u * v25];
        let v27 = v13 + varg2;
        let v28 = v27 + v0;
        let v29 = varg3[260u * v15 + 1u * v28];
        let v30 = v17 * varg5;
        let v31 = v20 * varg6;
        let v32 = v23 * varg6;
        let v33 = v17 * varg7;
        let v34 = v31 + v32;
        let v35 = v34 + v33;
        let v36 = v26 * varg6;
        let v37 = v29 * varg6;
        let v38 = v36 + v37;
        let vtemp = v38 + v33;
        let v39 = v35 + vtemp;
        let v40 = v39 * varg8;
        let v41 = v30 + varg9;
        let v42 = v41 + v40;
        let v43 = v42 * varg10;
        varg11[260u * v15 + 1u * v16] = v43;
            }

"""

# Create input data as a memoryview
n = 20
data = memoryview(bytearray(n * 4)).cast("i")
for i in range(n):
    data[i] = 1

# %% The short version, using memoryview

# The first arg is the input data, per binding
# The second arg are the ouput types, per binding
out = compute_with_buffers({0: data}, {1: (n, "i")}, new, n=n)

# The result is a dict matching the output types
# Select data from buffer at binding 1
result = out[1].tolist()
print(result)
# assert result == list(range(20))

# %% The short version, using numpy

# import numpy as np
#
# numpy_data = np.frombuffer(data, np.int32)
# out = compute_with_buffers({0: numpy_data}, {1: numpy_data.nbytes}, compute_shader, n=n)
# result = np.frombuffer(out[1], dtype=np.int32)
# print(result)


# %% The long version using the wgpu API

# Create device and shader object
device = wgpu.utils.get_default_device()
cshader = device.create_shader_module(code=new)

# Create buffer objects, input buffer is mapped.
buffer1 = device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.STORAGE)
buffer2 = device.create_buffer(
    size=data.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
)

# Setup layout and bindings
binding_layouts = [
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.read_only_storage,
        },
    },
    {
        "binding": 1,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
]
bindings = [
    {
        "binding": 0,
        "resource": {"buffer": buffer1, "offset": 0, "size": buffer1.size},
    },
    {
        "binding": 1,
        "resource": {"buffer": buffer2, "offset": 0, "size": buffer2.size},
    },
]

# Put everything together
bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

# Create and run the pipeline
compute_pipeline = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": cshader, "entry_point": "main"},
)
command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(compute_pipeline)
compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 elements not used
compute_pass.dispatch_workgroups(n, 1, 1)  # x y z
compute_pass.end()
device.queue.submit([command_encoder.finish()])

# Read result
# result = buffer2.read_data().cast("i")
out = device.queue.read_buffer(buffer2).cast("i")
result = out.tolist()
print(result)
# assert result == list(range(20))
