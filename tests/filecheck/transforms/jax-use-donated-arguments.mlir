// RUN: xdsl-opt %s -p jax-use-donated-arguments --split-input-file --verify-diagnostics | filecheck %s

func.func public @one_donation(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<2x4xf32> {tf.aliasing_output = 0 : i32}) -> (tensor<2x4xf32>) {
    %res = "test.op"() : () -> tensor<2x4xf32>
    return %res : tensor<2x4xf32>
  }

// CHECK: builtin.module {
// CHECK-NEXT:   func.func public @one_donation(%arg0 : tensor<2x3xf32>, %arg1 : tensor<3x4xf32>, %arg2 : tensor<2x4xf32>) -> tensor<2x4xf32> {
// CHECK-NEXT:      %res = "test.op"() : () -> tensor<2x4xf32>
// CHECK-NEXT:      %0 = bufferization.materialize_in_destination %res in %arg2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:      func.return %0 : tensor<2x4xf32>
// CHECK-NEXT:   }

func.func public @same_type_donation(%arg0: tensor<2x3xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<2x3xf32> {tf.aliasing_output = 0 : i32}, %arg2: tensor<2x3xf32> {tf.aliasing_output = 0 : i32}) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
    %res1 = "test.op"() : () -> tensor<2x3xf32>
    %res2 = "test.op"() : () -> tensor<2x3xf32>
    return %res1, %res2 : tensor<2x3xf32>, tensor<2x3xf32>
  }

// CHECK-NEXT:   func.func public @same_type_donation(%arg0 : tensor<2x3xf32>, %arg1 : tensor<2x3xf32>, %arg2 : tensor<2x3xf32> {tf.aliasing_output = 0 : i32}) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
// CHECK-NEXT:     %res1 = "test.op"() : () -> tensor<2x3xf32>
// CHECK-NEXT:     %res2 = "test.op"() : () -> tensor<2x3xf32>
// CHECK-NEXT:     %0 = bufferization.materialize_in_destination %res1 in %arg0 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     %1 = bufferization.materialize_in_destination %res2 in %arg1 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     func.return %0, %1 : tensor<2x3xf32>, tensor<2x3xf32>
// CHECK-NEXT:   }

func.func public @non_trivial_donation(%arg0: tensor<4x5xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<2x3xf32> {tf.aliasing_output = 0 : i32}, %arg2: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<4x5xf32>) {
    %res1 = "test.op"() : () -> tensor<2x3xf32>
    %res2 = "test.op"() : () -> tensor<2x3xf32>
    %res3 = "test.op"() : () -> tensor<4x5xf32>
    return %res1, %res2, %res3 : tensor<2x3xf32>, tensor<2x3xf32>, tensor<4x5xf32>
  }

// CHECK-NEXT:   func.func public @non_trivial_donation(%arg0 : tensor<4x5xf32>, %arg1 : tensor<2x3xf32>, %arg2 : tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<4x5xf32>) {
// CHECK-NEXT:      %res1 = "test.op"() : () -> tensor<2x3xf32>
// CHECK-NEXT:      %res2 = "test.op"() : () -> tensor<2x3xf32>
// CHECK-NEXT:      %res3 = "test.op"() : () -> tensor<4x5xf32>
// CHECK-NEXT:      %0 = bufferization.materialize_in_destination %res1 in %arg1 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:      %1 = bufferization.materialize_in_destination %res3 in %arg0 : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
// CHECK-NEXT:      func.return %0, %res2, %1 : tensor<2x3xf32>, tensor<2x3xf32>, tensor<4x5xf32>
// CHECK-NEXT:   }

func.func public @dont_double_buffer(%arg0 : tensor<2x3xf32>, %arg1 : tensor<2x3xf32>, %arg2 : tensor<2x3xf32> {"tf.aliasing_output" = 0 : i32}) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
  %res1 = "test.op"() : () -> tensor<2x3xf32>
  %res2 = "test.op"() : () -> tensor<2x3xf32>
  %0 = bufferization.materialize_in_destination %res1 in %arg0 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = bufferization.materialize_in_destination %res2 in %arg1 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %0, %1 : tensor<2x3xf32>, tensor<2x3xf32>
}

// CHECK-NEXT:   func.func public @dont_double_buffer(%arg0 : tensor<2x3xf32>, %arg1 : tensor<2x3xf32>, %arg2 : tensor<2x3xf32> {tf.aliasing_output = 0 : i32}) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
// CHECK-NEXT:     %res1 = "test.op"() : () -> tensor<2x3xf32>
// CHECK-NEXT:     %res2 = "test.op"() : () -> tensor<2x3xf32>
// CHECK-NEXT:     %0 = bufferization.materialize_in_destination %res1 in %arg0 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     %1 = bufferization.materialize_in_destination %res2 in %arg1 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     func.return %0, %1 : tensor<2x3xf32>, tensor<2x3xf32>
// CHECK-NEXT:   }

// CHECK-NEXT: }
