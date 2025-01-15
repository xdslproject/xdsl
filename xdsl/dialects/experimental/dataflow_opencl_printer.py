from typing import IO

from xdsl.dialects import builtin, func
from xdsl.printer import Printer


class OpenCLProgram:
    def __init__(self, program: builtin.ModuleOp, test=False):
        top_func = [
            func_op
            for func_op in program.ops
            if isinstance(func_op, func.FuncOp) and "TOP" in func_op.attributes
        ][0]
        node_calls = [
            node_call
            for node_call in top_func.body.block.ops
            if isinstance(node_call, func.Call)
        ]
        self.node_names = set(
            [node_call.callee.root_reference.data for node_call in node_calls]
        )

        self.all_node_funcs = [
            node_func
            for node_func in program.ops
            if isinstance(node_func, func.FuncOp)
            and node_func.sym_name.data in self.node_names
        ]

        self.host_pointers = []

        self.opencl_queues = self.get_command_queues()
        # for queue in self.opencl_queues:
        #    print(self.opencl_queues[queue])

        if test:
            self.program = f"""
const char *kernel_source_0 = R\"({self.get_conv_test(2, 0)})\";
const char *kernel_source_1 = R\"({self.get_conv_test(2, 1)})\";

cl_program program_0 = clCreateProgramWithSource(context, 1, &kernel_source_0, NULL, &err);
cl_program program_1 = clCreateProgramWithSource(context, 1, &kernel_source_1, NULL, &err);

err = clBuildProgram(program_0, 1, &device, NULL, NULL, NULL);
err = clBuildProgram(program_1, 1, &device, NULL, NULL, NULL);
            """
        else:
            self.program = """
    // Load kernel binary
    size_t binary_size;
    #ifdef SW_EMU
    unsigned char* binary = read_binary_file("nodes-sw_emu.xclbin", &binary_size);
    #elif HW_EMU
    unsigned char* binary = read_binary_file("nodes-hw_emu.xclbin", &binary_size);
    #endif

    cl_program program = clCreateProgramWithBinary(
            context,
            1,
            &device,
            &binary_size,
            (const unsigned char**)&binary,
            NULL,
            &status
        );
"""

        self.platform = """
cl_platform_id platform;
clGetPlatformIDs(1, &platform, NULL);


char platform_name[1024]; // Buffer to hold the platform name

// Get platform ID
err = clGetPlatformIDs(1, &platform, NULL);
if (err != CL_SUCCESS) {
    printf("Error getting platform ID: %d\\n", err);
    return 1;
}

// Get the platform name
err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
if (err != CL_SUCCESS) {
    printf("Error getting platform info: %d\\n", err);
    return 1;
}

// Print the platform name
printf("Platform Name: %s\\n", platform_name);
"""

        if test:
            self.device = """
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    """
        else:
            self.device = """
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device, NULL);
    """

        self.context = """
cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
"""
        self.opencl_kernels = self.get_kernels(test)
        # for kernel in self.opencl_kernels:
        #    print(self.opencl_kernels[kernel])

        self.buffers, self.buffer_names = self.get_all_buffers(self.host_pointers)
        # for node_func in self.all_node_funcs:
        #    node_name = node_func.sym_name.data
        #    node_buffers = self.buffers[node_name]
        #    for buf in node_buffers:
        #        print(buf)

        self.set_kernel_args = self.get_set_kernel_args(test)

    def get_ptr_arith_functions(self):
        _1D_to_3D = """
float ***_1D_to_3D(float *array, int D, int R, int C) {
    // Allocate memory for the 3D pointer-to-pointer structure (D x R x C)
    float ***ptr_array = (float ***)malloc(D * sizeof(float **));

    // Allocate memory for each slice (2D array)
    for (int i = 0; i < D; i++) {
        ptr_array[i] = (float **)malloc(R * sizeof(float *));

        // Allocate memory for each row (1D array of columns)
        for (int j = 0; j < R; j++) {
            ptr_array[i][j] = &array[(i * R * C) + (j * C)]; // Point to the correct memory in the 1D array
        }
    }

    // Return the 3D pointer structure
    return ptr_array;
}
"""

        _1D_to_4D = """
float ****_1D_to_4D(float *array, int D, int R, int C, int L) {
    // Allocate memory for the 4D pointer-to-pointer-to-pointer structure (D x R x C x L)
    float ****ptr_array = (float ****)malloc(D * sizeof(float ***));

    // Allocate memory for each depth slice (3D array)
    for (int i = 0; i < D; i++) {
        ptr_array[i] = (float ***)malloc(R * sizeof(float **));

        // Allocate memory for each row slice (2D array)
        for (int j = 0; j < R; j++) {
            ptr_array[i][j] = (float **)malloc(C * sizeof(float *));

            // Allocate memory for each column (1D array)
            for (int k = 0; k < C; k++) {
                ptr_array[i][j][k] = &array[(i * R * C * L) + (j * C * L) + (k * L)];
            }
        }
    }

    // Return the 4D pointer structure
    return ptr_array;
}
"""

        return _1D_to_3D, _1D_to_4D

    def get_command_queues(self) -> dict:
        opencl_queues = dict()
        for node_func in self.all_node_funcs:
            node_name = node_func.sym_name.data
            opencl_queues[node_name] = (
                f"cl_command_queue queue_{node_name} = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);\n"
            )
        opencl_queues["global"] = (
            "cl_command_queue queue_global = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);\n"
        )

        return opencl_queues

    def get_kernels(self, test=False) -> dict:
        opencl_kernels = dict()
        for idx, node_func in enumerate(self.all_node_funcs):
            node_name = node_func.sym_name.data
            if test:
                program_name = f"program_{idx}"
            else:
                program_name = "program"

            opencl_kernels[node_name] = (
                f'cl_kernel kernel_{node_name} = clCreateKernel({program_name}, "{node_name}", &err);\n'
            )

        return opencl_kernels

    def get_buffers_node(
        self, node_func: func.FuncOp, host_pointers: list[str]
    ) -> list:
        buffers = []
        buffer_names = []

        node_name = node_func.sym_name.data
        for arg_idx, arg in enumerate(node_func.function_type.inputs):
            if isinstance(arg, builtin.MemRefType):
                host_pointers.append(f"host_ptr_{node_name}_{arg_idx}")
                buffer_names.append(f"buf_{node_name}_{arg_idx}")
                buffers.append(
                    f"cl_mem {buffer_names[-1]} = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * ARRAY_SIZE, {host_pointers[-1]}, &err);\n"
                )

        return buffers, buffer_names

    def get_all_buffers(self, host_pointers: list[str]) -> dict:
        buffers = dict()
        buffer_names = []

        for node_func in self.all_node_funcs:
            node_name = node_func.sym_name.data
            buffers[node_name], buffer_node_names = self.get_buffers_node(
                node_func, host_pointers
            )
            buffer_names += buffer_node_names

        return buffers, buffer_names

    def get_set_kernel_args_node(self, node_func: func.FuncOp):
        node_name = node_func.sym_name.data
        # iter_buffers = iter(self.buffers[node_name])

        set_kernel_args = []
        for arg, arg_idx in enumerate(node_func.function_type.inputs):
            if isinstance(arg, builtin.MemRefType):
                set_kernel_args.append(
                    f"err = clSetKernelArg(kernel_{node_name}, {arg_idx}, sizeof(cl_mem), &buf_{node_name}_{arg_idx}); "
                )
            else:  # TODO: process scalars
                set_kernel_args.append(
                    f"err |= clSetKernelArg(kernel_{node_name}, {arg_idx}, sizeof(unsigned int), &DUMMY);"
                )

        return set_kernel_args

    def get_set_kernel_args(self, test=False) -> dict:
        set_kernel_args = dict()

        if test:
            set_kernel_args["sub_loop_node_0"] = [
                "err = clSetKernelArg(kernel_sub_loop_node_0, 0, sizeof(cl_mem), &buf_sub_loop_node_0_1);",
                "err = clSetKernelArg(kernel_sub_loop_node_0, 1, sizeof(cl_mem), &buf_sub_loop_node_0_2);",
                "err = clSetKernelArg(kernel_sub_loop_node_0, 2, sizeof(cl_mem), &buf_sub_loop_node_0_3);",
            ]
            set_kernel_args["sub_loop_node_1"] = [
                "err = clSetKernelArg(kernel_sub_loop_node_1, 0, sizeof(cl_mem), &buf_sub_loop_node_0_1);",
                "err = clSetKernelArg(kernel_sub_loop_node_1, 1, sizeof(cl_mem), &buf_sub_loop_node_0_2);",
                "err = clSetKernelArg(kernel_sub_loop_node_1, 2, sizeof(cl_mem), &buf_sub_loop_node_0_3);",
            ]
        else:
            for node_func in self.all_node_funcs:
                node_name = node_func.sym_name.data
                set_kernel_args[node_name] = self.get_set_kernel_args_node(node_func)

        return set_kernel_args

    def get_verify(test=False):
        if test:
            conv = """
  float *** host_input = _1D_to_3D(host_ptr_sub_loop_node_0_1, 8, 4, 4);
  float **** host_kernel = _1D_to_4D(host_ptr_sub_loop_node_0_2, 8, 8, 3, 3);
  float host_output[8][4][4];
  //float *** _output = _1D_to_3D(host_ptr_sub_loop_node_0_3, 8, 4, 4);

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        host_output[i][j][k] = 0;
        for (int l = 0; l < 8; l++) {
          for (int m = 0; m < 3; m++) {
            for (int n = 0; n < 3; n++) {
              int x = j * 2 + m - 1;
              int y = k * 2 + n - 1;
              if (x >= 0 && x < 4 && y >= 0 && y < 4) {
                host_output[i][j][k] += host_input[l][x][y] * host_kernel[i][l][m][n];
              }
            }
          }
        }
      }
    }
  }

  float *** device_output = _1D_to_3D(host_ptr_sub_loop_node_0_3, 8, 4, 4);
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        //printf("host: %f, device: %f\\n", host_output[i][j][k], device_output[i][j][k]);
        if(host_output[i][j][k] != device_output[i][j][k]) {
            printf("Read: %f at (%d,%d,%d). Expected: %f\\n", device_output[i][j][k], host_output[i][j][k]);
            return 1;
        }
      }
    }
  }
"""
        return conv

    #####################################################
    # TESTING
    #####################################################
    def get_conv_test(self, factor: int, node: int):
        outer = 8
        n_iters = outer / factor
        start = node * n_iters
        end = start + n_iters

        conv_test = f"""
__kernel void sub_loop_node_{node}(__global float _input[8][4][4],
                                          __global const float _kernel[8][8][3][3],
                                          __global float _output[8][4][4]) {{
  printf("----> RUNNING NODE {node}\\n");
  for (int i = {start}; i < {end}; i++) {{
    for (int j = 0; j < 4; j++) {{
      for (int k = 0; k < 4; k++) {{
        _output[i][j][k] = 0;
        for (int l = 0; l < 8; l++) {{
          for (int m = 0; m < 3; m++) {{
            for (int n = 0; n < 3; n++) {{
              int x = j * 2 + m - 1;
              int y = k * 2 + n - 1;
              if (x >= 0 && x < 4 && y >= 0 && y < 4) {{
                _output[i][j][k] += _input[l][x][y] * _kernel[i][l][m][n];
              }}
            }}
          }}
        }}
      }}
    }}
  }}
}}
        """

        return conv_test

    def get_main(self, test=False):
        main_str = f"""
#include <stdio.h>
#include <omp.h>
#include "token_queue/token_queue.h"
#include <stdlib.h>
#include <CL/cl.h>

#define ARRAY_SIZE 1000

// Error handling macro
#define CHECK_ERR(err) if (err != CL_SUCCESS) {{ fprintf(stderr, "OpenCL error: %d\\n", err); exit(1); }}

// Function to read binary file into a buffer
unsigned char* read_binary_file(const char* filename, size_t* binary_size) {{
    FILE* file = fopen(filename, "rb");
    if (!file) {{
        perror("Failed to open binary file");
        exit(1);
    }}
    fseek(file, 0, SEEK_END);
    *binary_size = ftell(file);
    rewind(file);
    unsigned char* binary = (unsigned char*)malloc(*binary_size);
    fread(binary, 1, *binary_size, file);
    fclose(file);
    return binary;
}}

{self.get_ptr_arith_functions()[0]}
{self.get_ptr_arith_functions()[1]}

int main() {{
    srand(42);

    // HOST POINTERS
    {';NEWLINE'.join(host_pointer for host_pointer in list(map(lambda x: "cl_float " + x + "[ARRAY_SIZE]", self.host_pointers)))};

    for (int i = 0; i < ARRAY_SIZE; i++) {{
        {'NEWLINE'.join(host_pointer for host_pointer in list(map(lambda x: x + '[i] = rand()%100;', self.host_pointers)))}
    }}
    cl_int err;
    // PLATFORM
    {self.platform}
    // DEVICE
    {self.device}
    // CONTEXT
    {self.context}
    // QUEUES
    {''.join(queue for queue in self.opencl_queues.values())}
    // BUFFERS
    {''.join(buffer for node_buffers in self.buffers.values() for buffer in node_buffers)}
    // PROGRAM
    cl_int status;
    {self.program}
    // KERNELS
    {''.join(kernel for kernel in self.opencl_kernels.values())}
    CHECK_ERR(err);
    {'NEWLINE'.join(set_kernel_args for node_set_kernel_args in self.set_kernel_args.values() for set_kernel_args in node_set_kernel_args)}
    CHECK_ERR(err);

    cl_mem buffers[] = {{{','.join(self.buffer_names)}}};
    clEnqueueMigrateMemObjects(queue_global, {len(self.buffer_names)}, buffers, 0, 0, NULL, NULL);
    clFinish(queue_global);

    // Each node that has predecessors has a queue per input token. Implementation wise
    // this means a token queue per pointer argument.
    TokenQueue tq_{self.all_node_funcs[1].sym_name.data} = {{
        .head = 0,
        .n_elems = 0,
    }};
    omp_init_lock(&tq_{self.all_node_funcs[1].sym_name.data}.lock);

    int n_top_iters = 1;

    #pragma omp parallel
    {{
        #pragma omp single
        {{
            #pragma omp task
            {{
                for(int i = 0; i < n_top_iters; i++) {{
                    Token token_{self.all_node_funcs[0].sym_name.data};
                    token_{self.all_node_funcs[0].sym_name.data}.tag = i;
                    //token_{self.all_node_funcs[0].sym_name.data}.start = start;
                    token_{self.all_node_funcs[0].sym_name.data}.stride = 1;
                    //token_{self.all_node_funcs[0].sym_name.data}.size = size_per_iter;
                    token_{self.all_node_funcs[0].sym_name.data}.consumed = false;

                    err = clEnqueueTask(queue_{self.all_node_funcs[0].sym_name.data}, kernel_{self.all_node_funcs[0].sym_name.data}, 0, NULL, &token_{self.all_node_funcs[0].sym_name.data}.event);

                    put(&tq_{self.all_node_funcs[1].sym_name.data}, &token_{self.all_node_funcs[0].sym_name.data});
                    clFinish(queue_{self.all_node_funcs[0].sym_name.data});
                }}
            }}

            #pragma omp task
            {{
                for(int i = 0; i < n_top_iters; i++) {{
                    while (is_empty(&tq_{self.all_node_funcs[1].sym_name.data})) {{}}
                    Token token = get_head(&tq_{self.all_node_funcs[1].sym_name.data});

                    cl_event in_tokens_kernel_{self.all_node_funcs[1].sym_name.data}[] = {{token.event}};

                    // Note: the synchronisation queue here allows the processing of the next
                    // token to proceed while the node runs.
                    clFinish(queue_{self.all_node_funcs[1].sym_name.data});
                    err = clEnqueueTask(queue_{self.all_node_funcs[1].sym_name.data}, kernel_{self.all_node_funcs[1].sym_name.data}, 1, in_tokens_kernel_{self.all_node_funcs[1].sym_name.data}, NULL);
                    CHECK_ERR(err);
                }}
            }}
        }}
        clFinish(queue_{self.all_node_funcs[1].sym_name.data});
    }}

    // Read back the result
    clEnqueueReadBuffer(queue_global, buf_sub_loop_node_0_3, CL_TRUE, 0, sizeof(float) * ARRAY_SIZE, host_ptr_sub_loop_node_0_3, 0, NULL, NULL);
    clFinish(queue_global);

    {self.get_verify()}

    return 0;
}}
""".replace("NEWLINE", "\n")

        return main_str


def print_dataflow_opencl(program: builtin.ModuleOp, output: IO[str]):
    printer = Printer(
        stream=output,
    )

    opencl_program = OpenCLProgram(program, test=True)

    main_str = opencl_program.get_main()
    printer.print(main_str)
