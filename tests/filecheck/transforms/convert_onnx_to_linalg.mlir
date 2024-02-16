// RUN: xdsl-opt -p convert-onnx-to-linalg %s | filecheck %s

%t0, %t1 = "test.op"() : () -> (tensor<3x2xf32>, tensor<3x2xf32>)
%res_add = onnx.Add(%t0, %t1) {onnx_node_name = "/Add"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
%t2 = "test.op"() : () -> (tensor<3x4xf32>)
%res_relu = onnx.Relu(%t2) {onnx_node_name = "/Relu"}: (tensor<3x4xf32>) -> tensor<3x4xf32>


// CHECK:       builtin.module {
// CHECK-NEXT:     %t0, %t1 = "test.op"() : () -> (tensor<3x2xf32>, tensor<3x2xf32>)
// CHECK-NEXT:     %res_add = tensor.empty() : tensor<3x2xf32>
// CHECK-NEXT:     %res_add_1 = linalg.add ins(%t0, %t1 : tensor<3x2xf32>, tensor<3x2xf32>) outs(%res_add : tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:     %t2 = "test.op"() : () -> tensor<3x4xf32>
// CHECK-NEXT:     %res_relu = tensor.empty() : tensor<3x4xf32>
// CHECK-NEXT:     %3 = linalg.generic {indexing_maps = [], iterator_types = []} ins(%t2 : tensor<3x4xf32>) outs(%res_relu : tensor<3x4xf32>) {
// CHECK-NEXT:     ^bb0(%in: tensor<3x4xf32>):
// CHECK-NEXT:      linalg.yield %in : tensor<3x4xf32>
// CHECK-NEXT:    } -> tensor<3x4xf32>
// CHECK-NEXT:  }

