// RUN: xdsl-opt -p convert-onnx-to-linalg %s | filecheck %s

%t0, %t1 = "test.op"() : () -> (tensor<3x2xf32>, tensor<3x2xf32>)
%res_add = onnx.Add(%t0, %t1) {onnx_node_name = "/Add"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
%t2 = "test.op"() : () -> (tensor<3x4xf32>)


// CHECK:       builtin.module {
// CHECK-NEXT:     %t0, %t1 = "test.op"() : () -> (tensor<3x2xf32>, tensor<3x2xf32>)
// CHECK-NEXT:     %res_add = tensor.empty() : tensor<3x2xf32>
// CHECK-NEXT:     %res_add_1 = linalg.add ins(%t0, %t1 : tensor<3x2xf32>, tensor<3x2xf32>) outs(%res_add : tensor<3x2xf32>) -> tensor<3x2xf32>

%res_relu = "onnx.Relu"(%t2) {onnx_node_name = "/Relu"}: (tensor<3x4xf32>) -> tensor<3x4xf32>

// CHECK-NEXT:     %res_relu = tensor.empty() : tensor<3x4xf32>
// CHECK-NEXT:     %zero = arith.constant 0.0 : f64
// CHECK-NEXT:     %3 = linalg.generic(%t2, %res_relu){indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>]}{
// CHECK-NEXT:    ^bb0(%a: f64, %b: f64):
// CHECK-NEXT:      %4 = arith.maxf %a, %zero : f64
// CHECK-NEXT:      linalg.yield %4 : f64
// CHECK-NEXT:    } -> tensor<3x4xf64>
// CHECK-NEXT:  }
