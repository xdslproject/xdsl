// RUN: xdsl-opt -p convert-onnx-to-linalg %s | filecheck %s

%t0, %t1 = "test.op"() : () -> (tensor<3x2xf32>, tensor<3x2xf32>)
%res_add = onnx.Add(%t0, %t1) {onnx_node_name = "/Add"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>


// CHECK:       builtin.module {
// CHECK-NEXT:     %t0, %t1 = "test.op"() : () -> (tensor<3x2xf32>, tensor<3x2xf32>)
// CHECK-NEXT:     %res_add = tensor.empty() : tensor<3x2xf32>
// CHECK-NEXT:     %res_add_1 = linalg.add ins(%t0, %t1 : tensor<3x2xf32>, tensor<3x2xf32>) outs(%res_add : tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:  }

