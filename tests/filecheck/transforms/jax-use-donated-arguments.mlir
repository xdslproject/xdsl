// RUN: xdsl-opt %s -p jax-use-donated-arguments --split-input-file --verify-diagnostics | filecheck %s

builtin.module {
func.func public @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<2x4xf32> {tf.aliasing_output = 0 : i32}) -> (tensor<2x4xf32>) {
    %0 = tensor.empty() : tensor<2x4xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }
}

// CHECK: builtin.module {
// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func public @main(%arg0 : tensor<2x3xf32>, %arg1 : tensor<3x4xf32>, %arg2 : tensor<2x4xf32> {"tf.aliasing_output" = 0 : i32}) -> tensor<2x4xf32> {
// CHECK-NEXT:      %0 = tensor.empty() : tensor<2x4xf32>
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:      %2 = bufferization.materialize_in_destination %1 in %arg2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:      func.return %2 : tensor<2x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }

builtin.module {
func.func public @main(%arg0: tensor<2x3xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<2x3xf32>, %arg2: tensor<4x5xf32> {tf.aliasing_output = 0 : i32}) -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<4x5xf32>) {
    %cst = arith.constant 0.000000e+00 : f32

    %0 = tensor.empty() : tensor<2x3xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x3xf32>) -> tensor<2x3xf32>

    %2 = tensor.empty() : tensor<2x3xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2x3xf32>) -> tensor<2x3xf32>

    %4 = tensor.empty() : tensor<4x5xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<4x5xf32>) -> tensor<4x5xf32>

    return %1, %3, %5 : tensor<2x3xf32>, tensor<2x3xf32>, tensor<4x5xf32>
  }
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func public @main(%arg0 : tensor<2x3xf32> {"tf.aliasing_output" = 0 : i32}, %arg1 : tensor<2x3xf32>, %arg2 : tensor<4x5xf32> {"tf.aliasing_output" = 0 : i32}) -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<4x5xf32>) {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %0 = tensor.empty() : tensor<2x3xf32>
// CHECK-NEXT:     %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     %2 = tensor.empty() : tensor<2x3xf32>
// CHECK-NEXT:     %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     %4 = tensor.empty() : tensor<4x5xf32>
// CHECK-NEXT:     %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<4x5xf32>) -> tensor<4x5xf32>
// CHECK-NEXT:     %6 = bufferization.materialize_in_destination %1 in %arg0 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     %7 = bufferization.materialize_in_destination %5 in %arg2 : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
// CHECK-NEXT:     func.return %6, %3, %7 : tensor<2x3xf32>, tensor<2x3xf32>, tensor<4x5xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK-NEXT: }
