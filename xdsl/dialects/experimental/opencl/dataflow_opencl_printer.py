import textwrap
from typing import IO

from xdsl.dialects import builtin, func
from xdsl.dialects.experimental.opencl import graph, prod_strings, test_strings, utils
from xdsl.printer import Printer


class OpenCLProgram:
    def __init__(self, program: builtin.ModuleOp, n_iters: int, test=False):
        self.n_iters = n_iters
        self.test = test

        self.graph = graph.Graph.generate_graph(program)

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
        self.scalars, self.scalar_names, self.scalar_arrays = self.get_all_scalars()
        self.set_kernel_args = self.get_set_kernel_args()

    def get_command_queues(self) -> dict:
        opencl_queues = dict()

        if self.test:
            for node_name in ["sub_loop_node_0", "sub_loop_node_1", "sub_loop_node_2"]:
                opencl_queues[node_name] = (
                    f"cl_command_queue queue_{node_name} = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);\n"
                )
        else:
            subnode_names = self.graph.get_subnode_names()
            for subnode_name in subnode_names:
                opencl_queues[subnode_name] = (
                    f"cl_command_queue queue_{subnode_name} = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);\n"
                )
        opencl_queues["global"] = (
            "cl_command_queue queue_global = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);\n"
        )

        return opencl_queues

    def get_kernels(self) -> dict:
        opencl_kernels = dict()
        if self.test:
            for idx, node_name in enumerate(
                ["sub_loop_node_0", "sub_loop_node_1", "sub_loop_node_2"]
            ):
                opencl_kernels[node_name] = (
                    f'cl_kernel kernel_{node_name} = clCreateKernel(program_{idx}, "{node_name}", &err);\n'
                )
        else:
            for subnode_name in self.graph.get_subnode_names():
                opencl_kernels[subnode_name] = (
                    f'cl_kernel kernel_{subnode_name} = clCreateKernel(program, "{subnode_name}", &err);\n'
                )

        return opencl_kernels

    def get_all_buffers(self, host_pointers: list[str]) -> dict:
        buffers = dict()
        buffer_names = []
        buffer_arrays = dict()

        for subnode_name in self.graph.get_subnode_names():
            subnode = self.graph.get_subnode_by_name(subnode_name)

            subnode_buf_indices = subnode.get_buf_arg_indices()
            buffers[subnode_name] = []
            buffer_arrays[subnode_name] = []

            for subnode_buf_idx in subnode_buf_indices:
                host_pointers.append(f"host_ptr_{subnode_name}_ARG{subnode_buf_idx}")
                buffer_name = f"buf_{subnode_name}_ARG{subnode_buf_idx}"
                buffer_names.append(buffer_name)

                buffer_arrays[subnode_name].append(
                    f"cl_mem {buffer_name}[{self.n_iters}];"
                )

                for iter in range(self.n_iters):
                    buffers[subnode_name].append(
                        f"{buffer_name}[{iter}] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * ARRAY_SIZE, host_ptr_{subnode_name}_ARG{subnode_buf_idx}[{iter}], &err);\n"
                    )

        return buffers, buffer_names, buffer_arrays

    def get_all_scalars(self):
        scalars = dict()
        scalar_names = []
        scalar_arrays = dict()

        for subnode_name in self.graph.get_subnode_names():
            subnode = self.graph.get_subnode_by_name(subnode_name)

            subnode_scalar_indices = subnode.get_scalar_arg_indices()
            scalars[subnode_name] = []
            scalar_arrays[subnode_name] = []

            for subnode_scalar_idx in subnode_scalar_indices:
                scalar_names.append(f"scalar_{subnode_name}_ARG{subnode_scalar_idx}")
                scalar_arrays[subnode_name].append(
                    f"float scalar_{subnode_name}_ARG{subnode_scalar_idx}[{self.n_iters}];"
                )

        return scalars, scalar_names, scalar_arrays

    def get_set_kernel_args_subnode(self, subnode_name: str, _iter: str):
        set_kernel_args_subnode = []

        buf_indices = self.graph.get_subnode_by_name(subnode_name).get_buf_arg_indices()
        scalar_indices = self.graph.get_subnode_by_name(
            subnode_name
        ).get_scalar_arg_indices()

        for buf_idx in buf_indices:
            set_kernel_args_subnode.append(
                f"err = clSetKernelArg(kernel_{subnode_name}, {buf_idx}, sizeof(cl_mem), &buf_{subnode_name}_ARG{buf_idx}[{_iter}]); "
            )
        for scalar_idx in scalar_indices:
            set_kernel_args_subnode.append(
                f"err |= clSetKernelArg(kernel_{subnode_name}, {scalar_idx}, sizeof(unsigned int), &scalar_{subnode_name}_ARG{scalar_idx}[{_iter}]);"  # TODO: process scalars
            )

        return set_kernel_args_subnode

    def get_all_kernel_args_list(self):
        all_kernel_args_list = []
        for subnode_name in self.graph.get_subnode_names():
            for _iter in range(self.n_iters):
                all_kernel_args_list += self.get_set_kernel_args_subnode(
                    subnode_name, str(_iter)
                )
        return all_kernel_args_list

    def get_set_kernel_args(self) -> dict:
        set_kernel_args = dict()

        for subnode_name in self.graph.get_subnode_names():
            set_kernel_args[subnode_name] = []
            for _iter in range(self.n_iters):
                set_kernel_args_subnode = self.get_set_kernel_args_subnode(
                    subnode_name, str(_iter)
                )
                set_kernel_args[subnode_name].append(set_kernel_args_subnode)

        return set_kernel_args

    def get_all_openmp_tasks(self):
        subnode_names = self.graph.get_subnode_names()

        openmp_tasks = []
        for subnode_name in subnode_names:
            openmp_tasks.append(self.generate_openmp_task(subnode_name))

        return openmp_tasks

    def generate_token_queue(self, subnode_name: str):
        return textwrap.dedent(f"""
            TokenQueue tq_{subnode_name} = {{
                .head = 0,
                .n_elems = 0,
            }};
            omp_init_lock(&tq_{subnode_name}.lock);""")

    def get_all_token_queues(self):
        subnode_names = self.graph.get_subnode_names()

        token_queues = []
        for subnode_name in subnode_names:
            token_queues.append(self.generate_token_queue(subnode_name))

        return token_queues

    def generate_openmp_task(self, subnode_name: str):
        out_token = textwrap.dedent(f"""
            Token token_{subnode_name};
            token_{subnode_name}.tag = iter;
            token_{subnode_name}.stride = 1;
            token_{subnode_name}.consumed = false;""")

        if self.graph.get_subnode_by_name(subnode_name).pred:
            wait_for_token = textwrap.dedent(f"""
                while (is_empty(&tq_{subnode_name})) {{}}""")

            process_in_dependencies = textwrap.dedent(f"""
                Token token_in_dep = get_head(&tq_{subnode_name});""")
        else:
            wait_for_token = ""
            process_in_dependencies = ""

        if self.graph.get_subnode_by_name(subnode_name).succ:
            process_out_dependencies = textwrap.dedent(f"""
                put(&tq_{subnode_name}, &token_{subnode_name});""")
        else:
            process_out_dependencies = ""

        openmp_task = textwrap.dedent(f"""
        #pragma omp task
        {{
            for(int iter = 0; iter < N_ITERS; iter++) {{
                {out_token}

                {wait_for_token}
                {process_in_dependencies}

                {'NEWLINE'.join(self.get_set_kernel_args_subnode(subnode_name, "iter"))}

                err = clEnqueueTask(queue_{subnode_name}, kernel_{subnode_name}, 0, NULL, &token_{subnode_name}.event);

                {process_out_dependencies}
                clFinish(queue_{subnode_name});
            }}
        }}""").replace("NEWLINE", "\n")

        return openmp_task

    def get_verify(self, iter_var: str):
        if self.test:
            conv = test_strings.get_verify(iter_var=iter_var)
        else:
            conv = ""
        return conv

    def get_main(self, test=False):
        main_str = ""
        main_str = textwrap.dedent(f"""
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

    // SCALARS
    {'NEWLINE'.join(scalar_array for node_scalar_arrays in self.scalar_arrays.values() for scalar_array in node_scalar_arrays)}
    {''.join(buffer for node_buffers in self.buffers.values() for buffer in node_buffers)}

    // PROGRAM
    cl_int status;
    {self.program}

    // KERNELS
    {''.join(kernel for kernel in self.opencl_kernels.values())}
    CHECK_ERR(err);
    {'NEWLINE'.join(self.get_all_kernel_args_list())}
    CHECK_ERR(err);

    for(int iter = 0; iter < N_ITERS; iter++) {{
        cl_mem buffers[] = {{{','.join(map(lambda x: x + "[iter]", self.buffer_names))}}};
        clEnqueueMigrateMemObjects(queue_global, {len(self.buffer_names)}, buffers, 0, 0, NULL, NULL);
    }}
    clFinish(queue_global);

    // Each node that has predecessors has a queue per input token. Implementation wise
    // this means a token queue per pointer argument.
    {'NEWLINE'.join(self.get_all_token_queues())}

    #pragma omp parallel
    {{
        #pragma omp single
        {{
            {'NEWLINE'.join(self.get_all_openmp_tasks())}
        }}
        clFinish(queue_{self.graph.get_subnode_names()[-1]});
    }}

    return 0;
}}
""").replace("NEWLINE", "\n")

        return main_str


def print_dataflow_opencl(program: builtin.ModuleOp, output: IO[str]):
    printer = Printer(
        stream=output,
    )

    opencl_program = OpenCLProgram(program, n_iters=2, test=False)

    main_str = opencl_program.get_main()
    printer.print(main_str)
