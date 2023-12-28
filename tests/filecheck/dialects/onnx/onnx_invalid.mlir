// RUN: xdsl-opt --verify-diagnostics --split-input-file %s | filecheck %s

// Non-broadcastable operands are not allowed.

builtin.module {
  %t0, %t1 = "test.op"() : () -> (tensor<2x4xf32>, tensor<3x2xf32>)

  // CHECK: Operation does not verify: operands have incompatible shapes: (2, 4) and (3, 2)
  %res_add =  "onnx.Add"(%t0, %t1) {onnx_node_name = "/Add"} : (tensor<2x4xf32>, tensor<3x2xf32>) -> tensor<2x4xf32>
}

// -----

builtin.module {
  %t0, %t1 = "test.op"() : () -> (tensor<2x4xf32>, tensor<1x4xf32>)

  // CHECK: Operation does not verify: result shape [2, 4] does not match result type tensor<1x4xf32>
  %res_sub =  "onnx.Sub"(%t0, %t1) {onnx_node_name = "/Sub"} : (tensor<2x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
}

// -----

builtin.module {
  %t0, %t1 = "test.op"() : () -> (f32, tensor<2x4xf32>)

  // CHECK: operand at position 0 does not verify!
  // CHECK: f32 should be of base attribute tensor
  %res_mul =  "onnx.Mul"(%t0, %t1) {onnx_node_name = "/Mul"} : (f32, tensor<2x4xf32>) -> tensor<2x4xf32>
}

// -----

builtin.module {
  %t0, %t1 = "test.op"() : () -> (tensor<2x4xf32>, tensor<2x4xi32>)

  // CHECK: operand at position 1 does not verify!
  // CHECK: attribute f32 expected from variable 'T', but got i32
  %res_div =  "onnx.Div"(%t0, %t1) {onnx_node_name = "/Div"} : (tensor<2x4xf32>, tensor<2x4xi32>) -> tensor<2x4xf32>
}
