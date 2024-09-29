// RUN: xdsl-opt %s -p jax-use-donated-arguments --split-input-file --verify-diagnostics | filecheck %s

builtin.module {
func.func public @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<2x4xf32> {tf.aliasing_output = 0 : i32}) -> (tensor<2x4xf32>) {
    %0 = tensor.empty() : tensor<2x4xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func public @main(%arg0 : tensor<2x3xf32>, %arg1 : tensor<3x4xf32>, %arg2 : tensor<2x4xf32> {"tf.aliasing_output" = 0 : i32}) -> tensor<2x4xf32> {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:     func.return %0 : tensor<2x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
