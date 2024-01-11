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

// -----

builtin.module {
  %t0 = "test.op"() : () -> (tensor<2x4xf32>)

  // CHECK: operand at position 1 does not verify!
  // CHECK: Operation does not verify: Mismatch between operand type and res type of onnx.Relu
  %res_relu =  "onnx.Relu"(%t0) {onnx_node_name = "/Relu"} : (tensor<2x4xf32>) -> tensor<3x4xf32>
}

// -----

builtin.module {
  %t0, %t1, %t2 = "test.op"() : () -> (tensor<2x4xf32>, tensor<3x2xf32>, tensor<3x2xf32>)

  // CHECK: Operation does not verify: operands have incompatible shapes: (2, 4) and (3, 2)
  %res_gemm =  "onnx.Gemm"(%t0, %t1, %t2) {onnx_node_name = "/Gemm"} : (tensor<2x4xf32>, tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<2x4xf32>
}

// -----

builtin.module {
  %t0, %t1, %t2 = "test.op"() : () -> (f32, tensor<2x4xf32>,tensor<2x4xf32>)

  // CHECK: operand at position 0 does not verify!
  // CHECK: f32 should be of base attribute tensor
  %res_gemm=  "onnx.Gemm"(%t0, %t1, %t2) {onnx_node_name = "/Gemm"} : (f32, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
}

// -----

builtin.module {
    %t0, %t1, %t2 = "test.op"(): () -> (tensor<5x2xf32>, tensor<2x1xf32>, tensor<5x4xf32>)

    //CHECK:  Operation does not verify: result shape [5, 4] does not match result type tensor<5x2xf32>
    %res_gemm = "onnx.Gemm"(%t0, %t1, %t2) : (tensor<5x2xf32>, tensor<2x1xf32>, tensor<5x4xf32>) -> tensor<5x2xf32>
}

// -----

builtin.module {
  %t0, %t1, %t2 = "test.op"() : () -> (tensor<2x4xf32>, tensor<2x4xi32>, tensor<2x4xf32>)

  // CHECK: operand at position 1 does not verify!
  // CHECK: attribute f32 expected from variable 'T', but got i32
  %res_gemm =  "onnx.Gemm"(%t0, %t1, %t2) {onnx_node_name = "/Gemm"} : (tensor<2x4xf32>, tensor<2x4xi32>, tensor<2x4xf32>) -> tensor<2x4xf32>

 }

 // -----

builtin.module {
  %t0, %t1, %t2 = "test.op"() : () -> (tensor<5x3x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>)
  // CHECK: Operation does not verify: tensor A should be a 2D tensor
  %res_gemm = "onnx.Gemm"(%t0, %t1, %t2) {onnx_node_name = "/Gemm"}: (tensor<5x3x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>) -> tensor<5x2xf32>
}

// -----

builtin.module {
  %t0, %t1, %t2 = "test.op"() : () -> (tensor<5x3xf32>, tensor<3x2x3xf32>, tensor<5x2xf32>)
  // CHECK: Operation does not verify: tensor B should be a 2D tensor
  %res_gemm = "onnx.Gemm"(%t0, %t1, %t2) {onnx_node_name = "/Gemm"}: (tensor<5x3xf32>, tensor<3x2x3xf32>, tensor<5x2xf32>) -> tensor<5x2xf32>
}

// -----

builtin.module {
  %t0, %t1, %t2 = "test.op"() : () -> (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2x7xf32>)
  // CHECK: Operation does not verify: tensor C should be a 1D tensor or 2D tensor
  %res_gemm = "onnx.Gemm"(%t0, %t1, %t2) {onnx_node_name = "/Gemm",  beta = 47.0 : f32}: (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2x7xf32>) -> tensor<5x3xf32>
}
