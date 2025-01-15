from typing import IO

from xdsl.dialects import builtin, func
from xdsl.dialects.experimental.opencl import prod_strings, test_strings, utils
from xdsl.printer import Printer


class OpenCLProgram:
    def __init__(self, program: builtin.ModuleOp, n_iters: int, test=False):
        self.n_iters = n_iters
        self.test = test
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

        if self.test:
            self.program = test_strings.program
        else:
            self.program = prod_strings.program

        self.platform = prod_strings.platform

        if self.test:
            self.device = test_strings.device
        else:
            self.device = prod_strings.device

        self.context = prod_strings.context
        self.opencl_kernels = self.get_kernels()
        self.buffers, self.buffer_names, self.buffer_arrays = self.get_all_buffers(
            self.host_pointers
        )
        self.set_kernel_args = self.get_set_kernel_args()

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

    def get_kernels(self) -> dict:
        opencl_kernels = dict()
        for idx, node_func in enumerate(self.all_node_funcs):
            node_name = node_func.sym_name.data
            if self.test:
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
        buffer_arrays = []

        node_name = node_func.sym_name.data
        for arg_idx, arg in enumerate(node_func.function_type.inputs):
            if isinstance(arg, builtin.MemRefType):
                host_pointers.append(f"host_ptr_{node_name}_{arg_idx}")
                buffer_names.append(f"buf_{node_name}_{arg_idx}")
                buffer_arrays.append(f"cl_mem {buffer_names[-1]}[{self.n_iters}];")
                for iter in range(self.n_iters):
                    buffers.append(
                        f"{buffer_names[-1]}[{iter}] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * ARRAY_SIZE, {host_pointers[-1]}[{iter}], &err);\n"
                    )

        return buffers, buffer_names, buffer_arrays

    def get_all_buffers(self, host_pointers: list[str]) -> dict:
        buffers = dict()
        buffer_names = []
        buffer_arrays = dict()

        for node_func in self.all_node_funcs:
            node_name = node_func.sym_name.data
            buffers[node_name], buffer_node_names, buffer_arrays[node_name] = (
                self.get_buffers_node(node_func, host_pointers)
            )
            buffer_names += buffer_node_names

        return buffers, buffer_names, buffer_arrays

    def get_set_kernel_args_node(self, node_func: func.FuncOp):
        node_name = node_func.sym_name.data

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

    def get_set_kernel_args(self) -> dict:
        set_kernel_args = dict()

        if self.test:
            set_kernel_args = test_strings.get_set_kernel_args(iter=0)
        else:
            for node_func in self.all_node_funcs:
                node_name = node_func.sym_name.data
                set_kernel_args[node_name] = self.get_set_kernel_args_node(node_func)

        return set_kernel_args

    def get_verify(self, iter_var: str):
        if self.test:
            conv = test_strings.get_verify(iter_var=iter_var)
        return conv

    def get_main(self, test=False):
        main_str = f"""
#include <stdio.h>
#include <omp.h>
#include "token_queue/token_queue.h"
#include <stdlib.h>
#include <CL/cl.h>

#define ARRAY_SIZE 1000
#define N_ITERS {self.n_iters}

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

{utils.get_ptr_arith_functions(3)}
{utils.get_ptr_arith_functions(4)}

int main() {{
    srand(42);

    // HOST POINTERS
    {';NEWLINE'.join(host_pointer for host_pointer in list(map(lambda x: "cl_float " + x + "[N_ITERS][ARRAY_SIZE]", self.host_pointers)))};

    for (int iter = 0; iter < N_ITERS; iter++) {{
        for (int i = 0; i < ARRAY_SIZE; i++) {{
            {'NEWLINE'.join(host_pointer for host_pointer in list(map(lambda x: x + '[iter][i] = rand()%100;', self.host_pointers)))}
        }}
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
    {'NEWLINE'.join(buffer_array for node_buffer_arrays in self.buffer_arrays.values() for buffer_array in node_buffer_arrays)}
    {''.join(buffer for node_buffers in self.buffers.values() for buffer in node_buffers)}
    // PROGRAM
    cl_int status;
    {self.program}
    // KERNELS
    {''.join(kernel for kernel in self.opencl_kernels.values())}
    CHECK_ERR(err);
    {'NEWLINE'.join(set_kernel_args for node_set_kernel_args in self.set_kernel_args.values() for set_kernel_args in node_set_kernel_args)}
    CHECK_ERR(err);

    for(int iter = 0; iter < N_ITERS; iter++) {{
        cl_mem buffers[] = {{{','.join(map(lambda x: x + "[iter]", self.buffer_names))}}};
        clEnqueueMigrateMemObjects(queue_global, {len(self.buffer_names)}, buffers, 0, 0, NULL, NULL);
    }}
    clFinish(queue_global);

    // Each node that has predecessors has a queue per input token. Implementation wise
    // this means a token queue per pointer argument.
    TokenQueue tq_{self.all_node_funcs[1].sym_name.data} = {{
        .head = 0,
        .n_elems = 0,
    }};
    omp_init_lock(&tq_{self.all_node_funcs[1].sym_name.data}.lock);

    #pragma omp parallel
    {{
        #pragma omp single
        {{
            #pragma omp task
            {{
                for(int i = 0; i < N_ITERS; i++) {{
                    Token token_{self.all_node_funcs[0].sym_name.data};
                    token_{self.all_node_funcs[0].sym_name.data}.tag = i;
                    //token_{self.all_node_funcs[0].sym_name.data}.start = start;
                    token_{self.all_node_funcs[0].sym_name.data}.stride = 1;
                    //token_{self.all_node_funcs[0].sym_name.data}.size = size_per_iter;
                    token_{self.all_node_funcs[0].sym_name.data}.consumed = false;

                    err = clSetKernelArg(kernel_sub_loop_node_0, 0, sizeof(cl_mem), &buf_sub_loop_node_0_1[i]);
                    err = clSetKernelArg(kernel_sub_loop_node_0, 1, sizeof(cl_mem), &buf_sub_loop_node_0_2[i]);
                    err = clSetKernelArg(kernel_sub_loop_node_0, 2, sizeof(cl_mem), &buf_sub_loop_node_0_3[i]);
                    err = clEnqueueTask(queue_{self.all_node_funcs[0].sym_name.data}, kernel_{self.all_node_funcs[0].sym_name.data}, 0, NULL, &token_{self.all_node_funcs[0].sym_name.data}.event);

                    put(&tq_{self.all_node_funcs[1].sym_name.data}, &token_{self.all_node_funcs[0].sym_name.data});
                    clFinish(queue_{self.all_node_funcs[0].sym_name.data});
                }}
            }}

            #pragma omp task
            {{
                for(int i = 0; i < N_ITERS; i++) {{
                    while (is_empty(&tq_{self.all_node_funcs[1].sym_name.data})) {{}}
                    Token token = get_head(&tq_{self.all_node_funcs[1].sym_name.data});

                    cl_event in_tokens_kernel_{self.all_node_funcs[1].sym_name.data}[] = {{token.event}};

                    // Note: the synchronisation queue here allows the processing of the next
                    // token to proceed while the node runs.
                    clFinish(queue_{self.all_node_funcs[1].sym_name.data});
                    err = clSetKernelArg(kernel_sub_loop_node_1, 0, sizeof(cl_mem), &buf_sub_loop_node_0_1[i]);
                    err = clSetKernelArg(kernel_sub_loop_node_1, 1, sizeof(cl_mem), &buf_sub_loop_node_0_2[i]);
                    err = clSetKernelArg(kernel_sub_loop_node_1, 2, sizeof(cl_mem), &buf_sub_loop_node_0_3[i]);
                    err = clEnqueueTask(queue_{self.all_node_funcs[1].sym_name.data}, kernel_{self.all_node_funcs[1].sym_name.data}, 1, in_tokens_kernel_{self.all_node_funcs[1].sym_name.data}, NULL);
                    CHECK_ERR(err);
                }}
            }}
        }}
        clFinish(queue_{self.all_node_funcs[1].sym_name.data});
    }}

    // VERIFY
    for (int iter = 0; iter < N_ITERS; iter++) {{
        // Read back the result
        clEnqueueReadBuffer(queue_global, buf_sub_loop_node_0_3[iter], CL_TRUE, 0, sizeof(float) * ARRAY_SIZE, host_ptr_sub_loop_node_0_3[iter], 0, NULL, NULL);
        clFinish(queue_global);

        {self.get_verify("iter")}
    }}

    return 0;
}}
""".replace("NEWLINE", "\n")

        return main_str


def print_dataflow_opencl(program: builtin.ModuleOp, output: IO[str]):
    printer = Printer(
        stream=output,
    )

    opencl_program = OpenCLProgram(program, n_iters=2, test=True)

    main_str = opencl_program.get_main()
    printer.print(main_str)
