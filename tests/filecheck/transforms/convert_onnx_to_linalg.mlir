// RUN: xdsl-opt -p convert-onnx-to-linalg %s | filecheck %s

%t0, %t1 = "test.op"() : () -> (tensor<3x2xf32>, tensor<3x2xf32>)
%res_add = onnx.Add(%t0, %t1) {onnx_node_name = "/Add"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>


// CHECK:       builtin.module {
// CHECK-NEXT:     %t0, %t1 = "test.op"() : () -> (tensor<3x2xf32>, tensor<3x2xf32>)
// CHECK-NEXT:     %res_add = tensor.empty() : tensor<3x2xf32>
// CHECK-NEXT:     %res_add_1 = linalg.add ins(%t0, %t1 : tensor<3x2xf32>, tensor<3x2xf32>) outs(%res_add : tensor<3x2xf32>) -> tensor<3x2xf32>

%t2 = "test.op"() : () -> (tensor<3x4xf64>)
%res_relu = "onnx.Relu"(%t2) {onnx_node_name = "/Relu"}: (tensor<3x4xf64>) -> tensor<3x4xf64>

// CHECK-NEXT:     %t2 = "test.op"() : () -> tensor<3x4xf64>
// CHECK-NEXT:     %res_relu = tensor.empty() : tensor<3x4xf64>
// CHECK-NEXT:     %res_relu_1 = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:     %res_relu_2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%t2 : tensor<3x4xf64>) outs(%res_relu : tensor<3x4xf64>) {
// CHECK-NEXT:     ^0(%0 : f64, %1 : f64):
// CHECK-NEXT:       %2 = arith.maximumf %0, %res_relu_1 : f64
// CHECK-NEXT:       linalg.yield %2 : f64
// CHECK-NEXT:    } -> tensor<3x4xf64>

%t3,%t4 = "test.op"(): () -> (tensor<20x2xf32>, tensor<2xi64>)
%res_reshape = "onnx.Reshape"(%t3, %t4) {onnx_node_name = "/Reshape"}: (tensor<20x2xf32>, tensor<2xi64>) -> tensor<1x40xf32>

// CHECK-NEXT: %t3, %t4 = "test.op"() : () -> (tensor<20x2xf32>, tensor<2xi64>)
// CHECK-NEXT: %res_reshape = tensor.reshape %t3(%t4) : (tensor<20x2xf32>, tensor<2xi64>) -> tensor<1x40xf32>

%t5, %t6, %t7 = "test.op"(): () -> (tensor<1x320xf32>, tensor<50x320xf32>, tensor<50xf32>)
%res_gemm= "onnx.Gemm"(%t5, %t6, %t7) {onnx_node_name = "/Gemm", "alpha" = 1.000000e+00 : f32, "beta" = 1.000000e+00 : f32, "transA" = 0 : si64, "transB" = 1 : si64}: (tensor<1x320xf32>, tensor<50x320xf32>, tensor<50xf32>) -> tensor<1x50xf32>

// CHECK-NEXT:  %t5, %t6, %t7 = "test.op"() : () -> (tensor<1x320xf32>, tensor<50x320xf32>, tensor<50xf32>)
// CHECK-NEXT:  %3 = tensor.empty() : tensor<320x50xf32>
// CHECK-NEXT:  %4 = linalg.transpose ins(%t6:tensor<50x320xf32>) outs(%3:tensor<320x50xf32>) permutation = [1, 0]
// CHECK-NEXT:  %res_gemm = tensor.empty() : tensor<1x50xf32>
// CHECK-NEXT:  %res_gemm_1 = linalg.matmul ins(%t5, %4 : tensor<1x320xf32>, tensor<320x50xf32>) outs(%res_gemm : tensor<1x50xf32>) -> tensor<1x50xf32>
// CHECK-NEXT:  %res_gemm_2 = linalg.add ins(%res_gemm_1, %t7 : tensor<1x50xf32>, tensor<50xf32>) outs(%res_gemm_1 : tensor<1x50xf32>) -> tensor<1x50xf32>



%t8, %t9, %t10 = "test.op"(): () -> (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>)
%res_gemm_1 = "onnx.Gemm"(%t8, %t9, %t10) {onnx_node_name = "/Gemm"}: (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>) -> tensor<5x2xf32>


// CHECK-NEXT:  %t8, %t9, %t10 = "test.op"() : () -> (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>)
// CHECK-NEXT:  %res_gemm_1 = tensor.empty() : tensor<5x2xf32>
// CHECK-NEXT:  %res_gemm_1_1 = linalg.matmul ins(%t8, %t9 : tensor<5x3xf32>, tensor<3x2xf32>) outs(%res_gemm_1 : tensor<5x2xf32>) -> tensor<5x2xf32>
// CHECK-NEXT:  %res_gemm_1_2 = linalg.add ins(%res_gemm_1_1, %t10 : tensor<5x2xf32>, tensor<5x2xf32>) outs(%res_gemm_1_1 : tensor<5x2xf32>) -> tensor<5x2xf32>


%t11, %t12, %t13 = "test.op"(): () -> (tensor<10x5xf32>, tensor<10x3xf32>, tensor<5x3xf32>)
%res_gemm_2 = "onnx.Gemm"(%t11, %t12, %t13) {onnx_node_name = "/Gemm", "alpha" = 0.500000e+00 : f32, "beta" = 0.500000e+00 : f32, "transA" = 1 : si64}: (tensor<10x5xf32>, tensor<10x3xf32>, tensor<5x3xf32>) -> tensor<5x3xf32>


// CHECK-NEXT:  %t11, %t12, %t13 = "test.op"() : () ->  (tensor<10x5xf32>, tensor<10x3xf32>, tensor<5x3xf32>)
// CHECK-NEXT:  %5 = tensor.empty() : tensor<5x10xf32>
// CHECK-NEXT:  %6 = linalg.transpose ins(%t11:tensor<10x5xf32>) outs(%5:tensor<5x10xf32>) permutation = [1, 0]
// CHECK-NEXT:  %7 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:  %8 = linalg.mul ins(%7, %6 : f32, tensor<5x10xf32>) outs(%6 : tensor<5x10xf32>) -> tensor<5x10xf32>
// CHECK-NEXT:  %9 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:  %10 = linalg.mul ins(%9, %t13 : f32, tensor<5x3xf32>) outs(%t13 : tensor<5x3xf32>) -> tensor<5x3xf32>
// CHECK-NEXT:  %res_gemm_2 = tensor.empty() : tensor<5x3xf32>
// CHECK-NEXT:  %res_gemm_2_1 = linalg.matmul ins(%8, %t12 : tensor<5x10xf32>, tensor<10x3xf32>) outs(%res_gemm_2 : tensor<5x3xf32>) -> tensor<5x3xf32>
// CHECK-NEXT:  %res_gemm_2_2 = linalg.add ins(%res_gemm_2_1, %10 : tensor<5x3xf32>, tensor<5x3xf32>) outs(%res_gemm_2_1 : tensor<5x3xf32>) -> tensor<5x3xf32>





%res_constant = "onnx.Constant"() {onnx_node_name = "/Constant", "value" = dense<1> : tensor<1xi64>}: () -> tensor<1xi64>
%res_constant_2 = "onnx.Constant"() {onnx_node_name = "/Constant", "value" = dense<2.0> : tensor<1x5xf32>} : () -> tensor<1x5xf32>

// CHECK-NEXT: %res_constant = ml_program.global_load_const @onnx_constant_1 : tensor<1xi64>
// CHECK-NEXT: %res_constant_2 = ml_program.global_load_const @onnx_constant_2 : tensor<1x5xf32>
// CHECK-NEXT: ml_program.global private @onnx_constant_1(dense<1> : tensor<1xi64>) : tensor<1xi64>
// CHECK-NEXT: ml_program.global private @onnx_constant_2(dense<2.000000e+00> : tensor<1x5xf32>) : tensor<1x5xf32>

// CHECK-NEXT:  }



