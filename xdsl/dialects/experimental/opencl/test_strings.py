def get_conv_test(factor: int, node: int):
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


def get_bn_test(factor: int, node: int):
    bn_test = f"""
    __kernel void sub_loop_node_{node}(__global float _input[8][4][4], __global const float scale[8],
         __global const float bias[8], __global const float mean[8], __global const float var[8],
         __global float _output[8][4][4]) {{
  printf("----> RUNNING NODE {node}\\n");
  for (int i = 0; i < 8; i++) {{
    for (int j = 0; j < 4; j++) {{
      for (int k = 0; k < 4; k++) {{
        // sqrtf(var[i] + 1e-5) removed, since it is not supported by the GPU
        _output[i][j][k] =
            (_input[i][j][k] - mean[i]) / (var[i] + 1e-5) * scale[i] +
            bias[i];
      }}
    }}
  }}
}}
"""

    return bn_test


####################################################################################################

program = f"""
const char *kernel_source_0 = R\"({get_conv_test(2, 0)})\";
const char *kernel_source_1 = R\"({get_conv_test(2, 1)})\";
const char *kernel_source_2 = R\"({get_bn_test(1, 2)})\";

cl_program program_0 = clCreateProgramWithSource(context, 1, &kernel_source_0, NULL, &err);
cl_program program_1 = clCreateProgramWithSource(context, 1, &kernel_source_1, NULL, &err);
cl_program program_2 = clCreateProgramWithSource(context, 1, &kernel_source_2, NULL, &err);

err = clBuildProgram(program_0, 1, &device, NULL, NULL, NULL);
err = clBuildProgram(program_1, 1, &device, NULL, NULL, NULL);
err = clBuildProgram(program_2, 1, &device, NULL, NULL, NULL);
"""

####################################################################################################

device = """
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
"""

####################################################################################################


def get_set_kernel_args(iter: str):
    set_kernel_args = dict()
    set_kernel_args["sub_loop_node_0"] = [
        f"err = clSetKernelArg(kernel_sub_loop_node_0, 0, sizeof(cl_mem), &buf_sub_loop_node_0_1[{iter}]);",
        f"err = clSetKernelArg(kernel_sub_loop_node_0, 1, sizeof(cl_mem), &buf_sub_loop_node_0_2[{iter}]);",
        f"err = clSetKernelArg(kernel_sub_loop_node_0, 2, sizeof(cl_mem), &buf_sub_loop_node_0_3[{iter}]);",
    ]
    set_kernel_args["sub_loop_node_1"] = [
        f"err = clSetKernelArg(kernel_sub_loop_node_1, 0, sizeof(cl_mem), &buf_sub_loop_node_0_1[{iter}]);",
        f"err = clSetKernelArg(kernel_sub_loop_node_1, 1, sizeof(cl_mem), &buf_sub_loop_node_0_2[{iter}]);",
        f"err = clSetKernelArg(kernel_sub_loop_node_1, 2, sizeof(cl_mem), &buf_sub_loop_node_0_3[{iter}]);",
    ]
    set_kernel_args["sub_loop_node_2"] = [
        f"err = clSetKernelArg(kernel_sub_loop_node_2, 0, sizeof(cl_mem), &buf_sub_loop_node_0_3[{iter}]);",
        f"err = clSetKernelArg(kernel_sub_loop_node_2, 1, sizeof(cl_mem), &buf_sub_loop_node_2_1[{iter}]);",
        f"err = clSetKernelArg(kernel_sub_loop_node_2, 2, sizeof(cl_mem), &buf_sub_loop_node_2_2[{iter}]);",
        f"err = clSetKernelArg(kernel_sub_loop_node_2, 3, sizeof(cl_mem), &buf_sub_loop_node_2_3[{iter}]);",
        f"err = clSetKernelArg(kernel_sub_loop_node_2, 4, sizeof(cl_mem), &buf_sub_loop_node_2_4[{iter}]);",
        f"err = clSetKernelArg(kernel_sub_loop_node_2, 5, sizeof(cl_mem), &buf_sub_loop_node_2_5[{iter}]);",
    ]

    return set_kernel_args


####################################################################################################


def get_verify(iter_var: str):
    #    verify = f"""
    #  float *** host_input = _1D_to_3D(host_ptr_sub_loop_node_0_1[{iter_var}], 8, 4, 4);
    #  float **** host_kernel = _1D_to_4D(host_ptr_sub_loop_node_0_2[{iter_var}], 8, 8, 3, 3);
    #  float host_output[8][4][4];
    #  //float *** _output = _1D_to_3D(host_ptr_sub_loop_node_0_3[{iter_var}], 8, 4, 4);
    #
    #  for (int i = 0; i < 8; i++) {{
    #    for (int j = 0; j < 4; j++) {{
    #      for (int k = 0; k < 4; k++) {{
    #        host_output[i][j][k] = 0;
    #        for (int l = 0; l < 8; l++) {{
    #          for (int m = 0; m < 3; m++) {{
    #            for (int n = 0; n < 3; n++) {{
    #              int x = j * 2 + m - 1;
    #              int y = k * 2 + n - 1;
    #              if (x >= 0 && x < 4 && y >= 0 && y < 4) {{
    #                host_output[i][j][k] += host_input[l][x][y] * host_kernel[i][l][m][n];
    #              }}
    #            }}
    #          }}
    #        }}
    #      }}
    #    }}
    #  }}
    #
    #  float *** device_output = _1D_to_3D(host_ptr_sub_loop_node_0_3[{iter_var}], 8, 4, 4);
    #  for (int i = 0; i < 8; i++) {{
    #    for (int j = 0; j < 4; j++) {{
    #      for (int k = 0; k < 4; k++) {{
    #        if(host_output[i][j][k] != device_output[i][j][k]) {{
    #            printf("(iter %d) Read: %f at (%d,%d,%d). Expected: %f\\n", {iter_var}, device_output[i][j][k], i, j, k, host_output[i][j][k]);
    #            return 1;
    #        }}
    #      }}
    #    }}
    #  }}
    # """

    # return verify
    return ""
