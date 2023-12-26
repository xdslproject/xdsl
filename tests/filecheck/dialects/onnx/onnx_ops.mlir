// RUN: XDSL_ROUNDTRIP

func.func @add(%arg0: tensor<1x2x6xf32>, %arg1: tensor<1x2x6xf32>) ->  tensor<1x2x6xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) {onnx_node_name = "/model.1/Add"} : (tensor<1x2x6xf32>, tensor<1x2x6xf32>) -> tensor<1x2x6xf32>
  return %0 : tensor<1x2x6xf32>
}

// CHECK-LABEL: func.func @add
// CHECK-SAME:      (%arg0 : tensor<1x2x6xf32>, %arg1 : tensor<1x2x6xf32>) -> tensor<1x2x6xf32>       
// CHECK:   %0 = onnx.Add(%arg0, %arg1) {"onnx_node_name" = "/model.1/Add"} : (tensor<1x2x6xf32>, tensor<1x2x6xf32>) -> tensor<1x2x6xf32>
