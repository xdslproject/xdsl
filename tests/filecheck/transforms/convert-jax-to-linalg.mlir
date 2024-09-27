// RUN: xdsl-opt %s -p convert-jax-to-linalg --split-input-file --verify-diagnostics | filecheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
builtin.module attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
func.func public @main(%arg0: tensor<2x3xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<3x4xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<2x4xf32> {mhlo.layout_mode = "default", tf.aliasing_output = 0 : i32}) -> (tensor<2x4xf32>) {
    %0 = tensor.empty() : tensor<2x4xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs(%1 : tensor<2x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
    } -> tensor<2x4xf32>
    return %2 : tensor<2x4xf32>
  }
}

// CHECK:      builtin.module attributes  {"mhlo.num_partitions" = 1 : i32, "mhlo.num_replicas" = 1 : i32} {
// CHECK-NEXT:   func.func public @main(%arg0 : tensor<2x3xf32> {"mhlo.layout_mode" = "default"}, %arg1 : tensor<3x4xf32> {"mhlo.layout_mode" = "default"}, %arg2 : tensor<2x4xf32> {"mhlo.layout_mode" = "default", "tf.aliasing_output" = 0 : i32}) -> tensor<2x4xf32> {
// CHECK-NEXT:     %0 = tensor.empty() : tensor<2x4xf32>
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %1 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:     %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs(%1 : tensor<2x4xf32>) {
// CHECK-NEXT:     ^0(%in : f32, %in_1 : f32, %out : f32):
// CHECK-NEXT:       %3 = arith.mulf %in, %in_1 : f32
// CHECK-NEXT:       %4 = arith.addf %out, %3 : f32
// CHECK-NEXT:       linalg.yield %4 : f32
// CHECK-NEXT:     } -> tensor<2x4xf32>
// CHECK-NEXT:     func.return %2 : tensor<2x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
