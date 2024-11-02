// RUN: XDSL_ROUNDTRIP

%t0, %t1 = "test.op"(): () -> (tensor<1x2x6xf32>, tensor<1x2x6xf32>)
%t2, %t3 = "test.op"(): () -> (tensor<3x2xf32>, tensor<1x2xf32>)
%t4, %t5 = "test.op"(): () -> (tensor<3x1x2xf32>, tensor<3x4x1xf32>)
%t6, %t7 = "test.op"(): () -> (tensor<1x5x1x3xf32>, tensor<2x1x6x3xf32>)
%t8 = "test.op"(): () -> (tensor<3x4xf32>)
%t9, %t10, %t11 = "test.op"(): () -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
%t12, %t13, %t14 = "test.op"(): () -> (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>)
%t15,%t16 = "test.op"(): () -> (tensor<48x256x64xf32>, tensor<3xi64>)
%t17,%t18 = "test.op"(): () -> (tensor<1x2x3x4x5xf32>, tensor<5xi64>)
%t19 = "test.op"(): () -> (tensor<10x10xf32>)
%t20,%t21,%t22 = "test.op"(): () ->  (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none)
%t23,%t24,%t25 = "test.op"(): () ->  (tensor<1x1x7x5xf32>, tensor<1x1x3x3xf32>, none)
%t26 = "test.op"(): () ->  (tensor<5x5x32x32xf32>)
%t27, %t28, %t29 = "test.op"(): () -> (tensor<1x320xf32>, tensor<50x320xf32>, tensor<50xf32>)

%res_add = "onnx.Add"(%t0, %t1) {onnx_node_name = "/Add"} : (tensor<1x2x6xf32>, tensor<1x2x6xf32>) -> tensor<1x2x6xf32>
// CHECK: %res_add = onnx.Add(%t0, %t1) {"onnx_node_name" = "/Add"} : (tensor<1x2x6xf32>, tensor<1x2x6xf32>) -> tensor<1x2x6xf32>

%res_sub = "onnx.Sub"(%t2, %t3) {onnx_node_name = "/Sub"}: (tensor<3x2xf32>, tensor<1x2xf32>) -> tensor<3x2xf32>
// CHECK: %res_sub = onnx.Sub(%t2, %t3) {"onnx_node_name" = "/Sub"} : (tensor<3x2xf32>, tensor<1x2xf32>) -> tensor<3x2xf32>

%res_mul = "onnx.Mul"(%t4, %t5) {onnx_node_name = "/Mul"}: (tensor<3x1x2xf32>, tensor<3x4x1xf32>) -> tensor<3x4x2xf32>
// CHECK: %res_mul = onnx.Mul(%t4, %t5) {"onnx_node_name" = "/Mul"} : (tensor<3x1x2xf32>, tensor<3x4x1xf32>) -> tensor<3x4x2xf32>

%res_div = "onnx.Div"(%t6, %t7) {onnx_node_name = "/Div"}: (tensor<1x5x1x3xf32>, tensor<2x1x6x3xf32>) -> tensor<2x5x6x3xf32>
// CHECK: %res_div = onnx.Div(%t6, %t7) {"onnx_node_name" = "/Div"} : (tensor<1x5x1x3xf32>, tensor<2x1x6x3xf32>) -> tensor<2x5x6x3xf32>

%res_relu = "onnx.Relu"(%t8) {onnx_node_name = "/Relu"}: (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK: %res_relu = onnx.Relu(%t8) {"onnx_node_name" = "/Relu"} : (tensor<3x4xf32>) -> tensor<3x4xf32>

%res_gemm = "onnx.Gemm"(%t9, %t10, %t11) {onnx_node_name = "/Gemm"}: (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK: %res_gemm = onnx.Gemm(%t9, %t10, %t11) {"onnx_node_name" = "/Gemm"} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

%res_gemm_1 = "onnx.Gemm"(%t12, %t13, %t14) {onnx_node_name = "/Gemm"}: (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>) -> tensor<5x2xf32>
// CHECK: %res_gemm_1 = onnx.Gemm(%t12, %t13, %t14) {"onnx_node_name" = "/Gemm"} : (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>) -> tensor<5x2xf32>

%res_reshape = "onnx.Reshape"(%t15, %t16) {onnx_node_name = "/Reshape", "allowzero" = 1 : i64}: (tensor<48x256x64xf32>, tensor<3xi64>) -> tensor<48x256x64xf32>
//CHECK: res_reshape = onnx.Reshape(%t15, %t16) {"onnx_node_name" = "/Reshape", "allowzero" = 1 : i64} : (tensor<48x256x64xf32>, tensor<3xi64>) -> tensor<48x256x64xf32>

%res_reshape_1 = "onnx.Reshape"(%t17, %t18) {onnx_node_name = "/Reshape"}: (tensor<1x2x3x4x5xf32>, tensor<5xi64>) -> tensor<1x120xf32>
//CHECK: %res_reshape_1 = onnx.Reshape(%t17, %t18) {"onnx_node_name" = "/Reshape"} : (tensor<1x2x3x4x5xf32>, tensor<5xi64>) -> tensor<1x120xf32>

%res_abs = "onnx.Abs"(%t19) {onnx_node_name = "/Abs"}: (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK: %res_abs = onnx.Abs(%t19) {"onnx_node_name" = "/Abs"} : (tensor<10x10xf32>) -> tensor<10x10xf32>

%res_conv = "onnx.Conv"(%t20, %t21, %t22) {onnx_node_name = "/Conv", "auto_pad" = "NOTSET", "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "strides" = [1 : i64, 1 : i64], "pads" = [1 : i64, 1 : i64, 1: i64, 1 : i64]}: (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x5x5xf32>
//CHECK: %res_conv = onnx.Conv(%t20, %t21, %t22) {"onnx_node_name" = "/Conv", "auto_pad" = "NOTSET", "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "strides" = [1 : i64, 1 : i64], "pads" = [1 : i64, 1 : i64, 1 : i64, 1 : i64]} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x5x5xf32>

%res_conv_1 = "onnx.Conv"(%t20, %t21, %t22) {onnx_node_name = "/Conv", "auto_pad" = "NOTSET", "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "strides" = [1 : i64, 1 : i64], "pads" = [0 : i64, 0 : i64, 0: i64, 0 : i64]}: (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>
//CHECK: %res_conv_1 = onnx.Conv(%t20, %t21, %t22) {"onnx_node_name" = "/Conv", "auto_pad" = "NOTSET", "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "strides" = [1 : i64, 1 : i64], "pads" = [0 : i64, 0 : i64, 0 : i64, 0 : i64]} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>

%res_conv_2 = "onnx.Conv"(%t20, %t21, %t22) {onnx_node_name = "/Conv", "auto_pad" = "SAME_LOWER", "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "strides" = [2 : i64, 2 : i64], "pads" = [0 : i64, 0 : i64, 0: i64, 0 : i64]}: (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>
//CHECK: %res_conv_2 = onnx.Conv(%t20, %t21, %t22) {"onnx_node_name" = "/Conv", "auto_pad" = "SAME_LOWER", "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "strides" = [2 : i64, 2 : i64], "pads" = [0 : i64, 0 : i64, 0 : i64, 0 : i64]} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>

%res_conv_3 = "onnx.Conv"(%t23, %t24, %t25) {onnx_node_name = "/Conv", "auto_pad" = "NOTSET", "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "strides" = [2 : i64, 2 : i64], "pads" = [1 : i64, 1 : i64, 1 : i64, 1 : i64]}: (tensor<1x1x7x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x4x3xf32>
//CHECK: %res_conv_3 = onnx.Conv(%t23, %t24, %t25) {"onnx_node_name" = "/Conv", "auto_pad" = "NOTSET", "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "strides" = [2 : i64, 2 : i64], "pads" = [1 : i64, 1 : i64, 1 : i64, 1 : i64]} : (tensor<1x1x7x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x4x3xf32>

%res_conv_4 = "onnx.Conv"(%t23, %t24, %t25) {onnx_node_name = "/Conv", "auto_pad" = "NOTSET", "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "strides" = [2 : i64, 2 : i64], "pads" = [0 : i64, 0 : i64, 0 : i64, 0 : i64]}: (tensor<1x1x7x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x2xf32>
//CHECK: %res_conv_4 = onnx.Conv(%t23, %t24, %t25) {"onnx_node_name" = "/Conv", "auto_pad" = "NOTSET", "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "strides" = [2 : i64, 2 : i64], "pads" = [0 : i64, 0 : i64, 0 : i64, 0 : i64]} : (tensor<1x1x7x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x2xf32>

%res_conv_5 = "onnx.Conv"(%t23, %t24, %t25) {onnx_node_name = "/Conv", "auto_pad" = "NOTSET", "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "strides" = [2 : i64, 2 : i64], "pads" = [1 : i64, 0 : i64, 1 : i64, 0 : i64]}: (tensor<1x1x7x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x4x2xf32>
//CHECK: %res_conv_5 = onnx.Conv(%t23, %t24, %t25) {"onnx_node_name" = "/Conv", "auto_pad" = "NOTSET", "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "strides" = [2 : i64, 2 : i64], "pads" = [1 : i64, 0 : i64, 1 : i64, 0 : i64]} : (tensor<1x1x7x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x4x2xf32>

%res_max_pool_single_out = "onnx.MaxPoolSingleOut"(%t26) {onnx_node_name = "/MaxPoolSingleOut", "auto_pad" = "VALID", "ceil_mode" = 0 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "pads" = [0 : i64, 0 : i64, 0 : i64, 0 : i64], "storage_order" = 0 : i64, "strides" = [1 : i64, 1 : i64]}: (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
//CHECK: %res_max_pool_single_out = onnx.MaxPoolSingleOut(%t26) {"onnx_node_name" = "/MaxPoolSingleOut", "auto_pad" = "VALID", "ceil_mode" = 0 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "pads" = [0 : i64, 0 : i64, 0 : i64, 0 : i64], "storage_order" = 0 : i64, "strides" = [1 : i64, 1 : i64]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>

"onnx.EntryPoint"() {onnx_node_name = "/EntryPoint", "func" = @main_graph} : () -> ()
//CHECK: "onnx.EntryPoint"() {"onnx_node_name" = "/EntryPoint", "func" = @main_graph} : () -> ()

%res_constant = onnx.Constant dense<1> : tensor<1xi64>
//CHECK: %res_constant = onnx.Constant dense<1> : tensor<1xi64>

%res_constant_1 = onnx.Constant dense<[5, 5, 16, 2]> : tensor<4xi64>
//CHECK: %res_constant_1 = onnx.Constant dense<[5, 5, 16, 2]> : tensor<4xi64>

%res_gemm_2 = "onnx.Gemm"(%t27, %t28, %t29) {onnx_node_name = "/Gemm", "alpha" = 1.000000e+00 : f32, "beta" = 1.000000e+00 : f32, "transA" = 0 : si64, "transB" = 1 : si64}: (tensor<1x320xf32>, tensor<50x320xf32>, tensor<50xf32>) -> tensor<1x50xf32>
// CHECK:  %res_gemm_2 = onnx.Gemm(%t27, %t28, %t29) {"onnx_node_name" = "/Gemm", "alpha" = 1.000000e+00 : f32, "beta" = 1.000000e+00 : f32, "transA" = 0 : si64, "transB" = 1 : si64} : (tensor<1x320xf32>, tensor<50x320xf32>, tensor<50xf32>) -> tensor<1x50xf32>

%res_matmul = "onnx.MatMul"(%t9, %t10) {onnx_node_name = "/MatMul"}: (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK: %res_matmul = onnx.MatMul(%t9, %t10) {"onnx_node_name" = "/MatMul"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

%res_transpose = "onnx.Transpose"(%t8) {onnx_node_name = "/Transpose", "perm" = [1 : i64, 0 : i64]}: (tensor<3x4xf32>) -> tensor<4x3xf32>
// CHECK:  %res_transpose = onnx.Transpose(%t8) {"onnx_node_name" = "/Transpose", "perm" = [1 : i64, 0 : i64]} : (tensor<3x4xf32>) -> tensor<4x3xf32>

%res_squeeze = "onnx.Squeeze"(%t0) {onnx_node_name = "/Squeeze", "axes" = 0}: (tensor<1x2x6xf32>) -> tensor<2x6xf32>
// CHECK: %res_squeeze = onnx.Squeeze(%t0) {"onnx_node_name" = "/Squeeze", "axes" = 0 : i64} : (tensor<1x2x6xf32>) -> tensor<2x6xf32>

%res_sigmoid = "onnx.Sigmoid"(%t8) {onnx_node_name = "/Sigmoid"}: (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK: %res_sigmoid = onnx.Sigmoid(%t8) {"onnx_node_name" = "/Sigmoid"} : (tensor<3x4xf32>) -> tensor<3x4xf32>
