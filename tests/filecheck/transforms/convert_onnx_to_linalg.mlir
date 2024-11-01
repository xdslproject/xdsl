// RUN: xdsl-opt --print-reduced-precision-fp -p convert-onnx-to-linalg %s | filecheck %s

// CHECK:       builtin.module {

%t0, %t1 = "test.op"() : () -> (tensor<3x2xf32>, tensor<3x2xf32>)
%res_add = onnx.Add(%t0, %t1) {onnx_node_name = "/Add"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
%res_sub = onnx.Sub(%t0, %t1) {onnx_node_name = "/Sub"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>

// CHECK-NEXT:     %t0, %t1 = "test.op"() : () -> (tensor<3x2xf32>, tensor<3x2xf32>)
// CHECK-NEXT:     %res_add = tensor.empty() : tensor<3x2xf32>
// CHECK-NEXT:     %res_add_1 = linalg.add ins(%t0, %t1 : tensor<3x2xf32>, tensor<3x2xf32>) outs(%res_add : tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:     %res_sub = tensor.empty() : tensor<3x2xf32>
// CHECK-NEXT:     %res_sub_1 = linalg.sub ins(%t0, %t1 : tensor<3x2xf32>, tensor<3x2xf32>) outs(%res_sub : tensor<3x2xf32>) -> tensor<3x2xf32>


%t2 = "test.op"() : () -> (tensor<3x4xf32>)
%res_relu = "onnx.Relu"(%t2) {onnx_node_name = "/Relu"}: (tensor<3x4xf32>) -> tensor<3x4xf32>

// CHECK-NEXT:     %t2 = "test.op"() : () -> tensor<3x4xf32>
// CHECK-NEXT:     %res_relu = tensor.empty() : tensor<3x4xf32>
// CHECK-NEXT:     %res_relu_1 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %res_relu_2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%t2 : tensor<3x4xf32>) outs(%res_relu : tensor<3x4xf32>) {
// CHECK-NEXT:     ^0(%0 : f32, %1 : f32):
// CHECK-NEXT:       %2 = arith.maximumf %0, %res_relu_1 : f32
// CHECK-NEXT:       linalg.yield %2 : f32
// CHECK-NEXT:    } -> tensor<3x4xf32>

%t27 = "test.op"() : () -> (tensor<3x4xf64>)
%res_relu_3 = "onnx.Relu"(%t27) {onnx_node_name = "/Relu"}: (tensor<3x4xf64>) -> tensor<3x4xf64>

// CHECK-NEXT:   %t27 = "test.op"() : () -> tensor<3x4xf64>
// CHECK-NEXT:   %res_relu_3 = tensor.empty() : tensor<3x4xf64>
// CHECK-NEXT:   %res_relu_4 = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %res_relu_5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%t27 : tensor<3x4xf64>) outs(%res_relu_3 : tensor<3x4xf64>) {
// CHECK-NEXT:   ^1(%3 : f64, %4 : f64):
// CHECK-NEXT:     %5 = arith.maximumf %3, %res_relu_4 : f64
// CHECK-NEXT:     linalg.yield %5 : f64
// CHECK-NEXT:   } -> tensor<3x4xf64>


%t3,%t4 = "test.op"(): () -> (tensor<20x2xf32>, tensor<2xi64>)
%res_reshape = "onnx.Reshape"(%t3, %t4) {onnx_node_name = "/Reshape"}: (tensor<20x2xf32>, tensor<2xi64>) -> tensor<1x40xf32>

// CHECK-NEXT: %t3, %t4 = "test.op"() : () -> (tensor<20x2xf32>, tensor<2xi64>)
// CHECK-NEXT: %res_reshape = tensor.reshape %t3(%t4) : (tensor<20x2xf32>, tensor<2xi64>) -> tensor<1x40xf32>

%t5, %t6, %t7 = "test.op"(): () -> (tensor<1x320xf32>, tensor<50x320xf32>, tensor<50xf32>)
%res_gemm= "onnx.Gemm"(%t5, %t6, %t7) {onnx_node_name = "/Gemm", "alpha" = 1.000000e+00 : f32, "beta" = 1.000000e+00 : f32, "transA" = 0 : si64, "transB" = 1 : si64}: (tensor<1x320xf32>, tensor<50x320xf32>, tensor<50xf32>) -> tensor<1x50xf32>

// CHECK-NEXT:  %t5, %t6, %t7 = "test.op"() : () -> (tensor<1x320xf32>, tensor<50x320xf32>, tensor<50xf32>)
// CHECK-NEXT:  %6 = tensor.empty() : tensor<320x50xf32>
// CHECK-NEXT:  %7 = linalg.transpose ins(%t6:tensor<50x320xf32>) outs(%6:tensor<320x50xf32>) permutation = [1, 0]
// CHECK-NEXT:  %res_gemm = tensor.empty() : tensor<1x50xf32>
// CHECK-NEXT:  %res_gemm_1 = linalg.matmul ins(%t5, %7 : tensor<1x320xf32>, tensor<320x50xf32>) outs(%res_gemm : tensor<1x50xf32>) -> tensor<1x50xf32>
// CHECK-NEXT:  %res_gemm_2 = linalg.add ins(%res_gemm_1, %t7 : tensor<1x50xf32>, tensor<50xf32>) outs(%res_gemm_1 : tensor<1x50xf32>) -> tensor<1x50xf32>



%t8, %t9, %t10 = "test.op"(): () -> (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>)
%res_gemm_1 = "onnx.Gemm"(%t8, %t9, %t10) {onnx_node_name = "/Gemm"}: (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>) -> tensor<5x2xf32>


// CHECK-NEXT:  %t8, %t9, %t10 = "test.op"() : () -> (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>)
// CHECK-NEXT:  %res_gemm_3 = tensor.empty() : tensor<5x2xf32>
// CHECK-NEXT:  %res_gemm_4 = linalg.matmul ins(%t8, %t9 : tensor<5x3xf32>, tensor<3x2xf32>) outs(%res_gemm_3 : tensor<5x2xf32>) -> tensor<5x2xf32>
// CHECK-NEXT:  %res_gemm_5 = linalg.add ins(%res_gemm_4, %t10 : tensor<5x2xf32>, tensor<5x2xf32>) outs(%res_gemm_4 : tensor<5x2xf32>) -> tensor<5x2xf32>


%t11, %t12, %t13 = "test.op"(): () -> (tensor<10x5xf32>, tensor<10x3xf32>, tensor<5x3xf32>)
%res_gemm_2 = "onnx.Gemm"(%t11, %t12, %t13) {onnx_node_name = "/Gemm", "alpha" = 0.500000e+00 : f32, "beta" = 0.500000e+00 : f32, "transA" = 1 : si64}: (tensor<10x5xf32>, tensor<10x3xf32>, tensor<5x3xf32>) -> tensor<5x3xf32>


// CHECK-NEXT:  %t11, %t12, %t13 = "test.op"() : () ->  (tensor<10x5xf32>, tensor<10x3xf32>, tensor<5x3xf32>)
// CHECK-NEXT:  %8 = tensor.empty() : tensor<5x10xf32>
// CHECK-NEXT:  %9 = linalg.transpose ins(%t11:tensor<10x5xf32>) outs(%8:tensor<5x10xf32>) permutation = [1, 0]
// CHECK-NEXT:  %10 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:  %11 = linalg.mul ins(%10, %9 : f32, tensor<5x10xf32>) outs(%9 : tensor<5x10xf32>) -> tensor<5x10xf32>
// CHECK-NEXT:  %12 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:  %13 = linalg.mul ins(%12, %t13 : f32, tensor<5x3xf32>) outs(%t13 : tensor<5x3xf32>) -> tensor<5x3xf32>
// CHECK-NEXT:  %res_gemm_6 = tensor.empty() : tensor<5x3xf32>
// CHECK-NEXT:  %res_gemm_7 = linalg.matmul ins(%11, %t12 : tensor<5x10xf32>, tensor<10x3xf32>) outs(%res_gemm_6 : tensor<5x3xf32>) -> tensor<5x3xf32>
// CHECK-NEXT:  %res_gemm_8 = linalg.add ins(%res_gemm_7, %13 : tensor<5x3xf32>, tensor<5x3xf32>) outs(%res_gemm_7 : tensor<5x3xf32>) -> tensor<5x3xf32>

%t26 = "test.op"(): () ->  (tensor<1x16x14x14xf32>)
%res_max_pool_single_out = "onnx.MaxPoolSingleOut"(%t26) {onnx_node_name = "/MaxPoolSingleOut", "auto_pad" = "NOTSET", "ceil_mode" = 0 : si64, "kernel_shape" = [3 : i64, 3 : i64],  "dilations" = [1 : i64, 1 : i64],  "pads" = [0 : i64, 0 : i64, 0 : i64, 0 : i64],  "storage_order" = 0 : si64, strides = [3 : i64, 3 : i64]} : (tensor<1x16x14x14xf32>) -> tensor<1x16x4x4xf32>

// CHECK-NEXT:   %t26 = "test.op"() : () -> tensor<1x16x14x14xf32>
// CHECK-NEXT:  %res_max_pool_single_out = tensor.empty() : tensor<3x3xf32>
// CHECK-NEXT:  %res_max_pool_single_out_1 = tensor.empty() : tensor<1x16x4x4xf32>
// CHECK-NEXT:  %res_max_pool_single_out_2 = arith.constant -1.000000e+308 : f64
// CHECK-NEXT:  %res_max_pool_single_out_3 = linalg.fill ins(%res_max_pool_single_out_2 : f64) outs(%res_max_pool_single_out_1 : tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32>
// CHECK-NEXT:  %res_max_pool_single_out_4 = linalg.pooling_nchw_max {"dilations" = dense<1> : tensor<2xi64>, "strides" = dense<3> : tensor<2xi64>} ins(%t26, %res_max_pool_single_out : tensor<1x16x14x14xf32>, tensor<3x3xf32>) outs(%res_max_pool_single_out_3 : tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32>

%t20, %t21, %t22 = "test.op"() : () -> (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none)
%res_conv_2 = "onnx.Conv"(%t20, %t21, %t22) {onnx_node_name = "/Conv", "auto_pad" = "NOTSET", "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "strides" = [1 : i64, 1 : i64], "pads" = [0 : i64, 0 : i64, 0: i64, 0 : i64]}: (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>

// CHECK-NEXT:  %t20, %t21, %t22 = "test.op"() : () -> (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none)
// CHECK-NEXT:  %res_conv = tensor.empty() : tensor<1x1x3x3xf32>
// CHECK-NEXT:  %res_conv_1 = linalg.conv_2d_nchw_fchw {"dilations" = dense<1> : tensor<2xi64>, "strides" = dense<1> : tensor<2xi64>} ins(%t20, %t21 : tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>) outs(%res_conv : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>

%t23, %t24, %t25 = "test.op"() : () -> (tensor<1x8x14x14xf32>, tensor<16x8x5x5xf32>, tensor<16xf32>)
%res_conv_3 = "onnx.Conv"(%t23, %t24, %t25) {onnx_node_name = "/Conv", "auto_pad" = "SAME_UPPER", "group" = 1 : i64, "kernel_shape" = [5 : i64, 5 : i64], "dilations" = [1 : i64, 1 : i64], "strides" = [1 : i64, 1 : i64], "pads" = [0 : i64, 0 : i64, 0: i64, 0 : i64]} : (tensor<1x8x14x14xf32>, tensor<16x8x5x5xf32>, tensor<16xf32>) -> tensor<1x16x14x14xf32>

// CHECK-NEXT:   %t23, %t24, %t25 = "test.op"() : () -> (tensor<1x8x14x14xf32>, tensor<16x8x5x5xf32>, tensor<16xf32>)
// CHECK-NEXT:   %res_conv_2 = tensor.empty() : tensor<1x16x14x14xf32>
// CHECK-NEXT:   %res_conv_3 = linalg.conv_2d_nchw_fchw {"dilations" = dense<1> : tensor<2xi64>, "strides" = dense<1> : tensor<2xi64>} ins(%t23, %t24 : tensor<1x8x14x14xf32>, tensor<16x8x5x5xf32>) outs(%res_conv_2 : tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32>
// CHECK-NEXT:   %res_conv_4 = linalg.add ins(%t25 : tensor<16xf32>) outs(%res_conv_3 : tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32>

%res_constant = "onnx.Constant"() {onnx_node_name = "/Constant", "value" = dense<1> : tensor<1xi64>}: () -> tensor<1xi64>
%res_constant_2 = "onnx.Constant"() {onnx_node_name = "/Constant", "value" = dense<2.0> : tensor<1x5xf32>} : () -> tensor<1x5xf32>

// CHECK-NEXT: %res_constant = ml_program.global_load_const @onnx_constant_1 : tensor<1xi64>
// CHECK-NEXT: %res_constant_1 = ml_program.global_load_const @onnx_constant_2 : tensor<1x5xf32>
// CHECK-NEXT: ml_program.global private @onnx_constant_1(dense<1> : tensor<1xi64>) : tensor<1xi64>
// CHECK-NEXT: ml_program.global private @onnx_constant_2(dense<2.000000e+00> : tensor<1x5xf32>) : tensor<1x5xf32>

// CHECK-NEXT:  }
