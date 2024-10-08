builtin.module {
  builtin.module {
    func.func public @main(%arg0 : tensor<2x3xf32>, %arg1 : tensor<3x4xf32>, %arg2 : tensor<2x4xf32> {"tf.aliasing_output" = 0 : i32}) -> tensor<2x4xf32> {
      %0 = tensor.empty() : tensor<2x4xf32>
      %cst = arith.constant 0.000000e+00 : f32
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>
      %2 = bufferization.materialize_in_destination %1 in %arg2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      func.return %2 : tensor<2x4xf32>
    }
  }
  builtin.module {
    func.func public @main(%arg0 : tensor<2x3xf32> {"tf.aliasing_output" = 0 : i32}, %arg1 : tensor<2x3xf32>, %arg2 : tensor<4x5xf32> {"tf.aliasing_output" = 0 : i32}) -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<4x5xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = tensor.empty() : tensor<2x3xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x3xf32>) -> tensor<2x3xf32>
      %2 = tensor.empty() : tensor<2x3xf32>
      %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2x3xf32>) -> tensor<2x3xf32>
      %4 = tensor.empty() : tensor<4x5xf32>
      %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<4x5xf32>) -> tensor<4x5xf32>
      %6 = bufferization.materialize_in_destination %1 in %arg0 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
      %7 = bufferization.materialize_in_destination %5 in %arg2 : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
      func.return %6, %3, %7 : tensor<2x3xf32>, tensor<2x3xf32>, tensor<4x5xf32>
    }
  }
}
