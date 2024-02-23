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

%res_constant = "onnx.Constant"() {onnx_node_name = "/Constant", "value" = dense<1> : tensor<1xi64>}: () -> tensor<1xi64>

// CHECK-NEXT: ml_program.global private @global_constant(dense<1> : tensor<1xi64>) : tensor<1xi64>
// CHECK-NEXT: ml_program.global_load_const @global_constant : tensor<1xi64>
// CHECK-NEXT:  }


