// RUN: xdsl-opt %s -p jax-use-donated-arguments --split-input-file | mlir-opt --eliminate-empty-tensors --one-shot-bufferize=bufferize-function-boundaries | filecheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
builtin.module {
  func.func public @main(%arg0: tensor<2x3xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<3x4xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<2x4xf32> {mhlo.layout_mode = "default", tf.aliasing_output = 0 : i32}) -> tensor<2x4xf32> {
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

// CHECK:      #map = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-NEXT: #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-NEXT: #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-NEXT: #map3 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT: module {
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func public @main(%arg0: memref<2x3xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default"}, %arg1: memref<3x4xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default"}, %arg2: memref<2x4xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default", tf.aliasing_output = 0 : i32}) -> memref<2x4xf32, strided<[?, ?], offset: ?>> {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     linalg.fill ins(%cst : f32) outs(%arg2 : memref<2x4xf32, strided<[?, ?], offset: ?>>)
// CHECK-NEXT:     linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<2x3xf32, strided<[?, ?], offset: ?>>, memref<3x4xf32, strided<[?, ?], offset: ?>>) outs(%arg2 : memref<2x4xf32, strided<[?, ?], offset: ?>>) {
// CHECK-NEXT:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK-NEXT:       %0 = arith.mulf %in, %in_0 : f32
// CHECK-NEXT:       %1 = arith.addf %out, %0 : f32
// CHECK-NEXT:       linalg.yield %1 : f32
// CHECK-NEXT:     }
// CHECK-NEXT:     memref.copy %arg2, %arg2 : memref<2x4xf32, strided<[?, ?], offset: ?>> to memref<2x4xf32, strided<[?, ?], offset: ?>>
// CHECK-NEXT:     return %arg2 : memref<2x4xf32, strided<[?, ?], offset: ?>>
// CHECK-NEXT:   }
// CHECK-NEXT: }

#map3 = affine_map<(d0, d1) -> (d0, d1)>
builtin.module {
  func.func public @main(%arg0: tensor<5x2xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<2x5xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<5x5xf32> {mhlo.layout_mode = "default", tf.aliasing_output = 2 : i32}, %arg3: tensor<2x2xf32> {mhlo.layout_mode = "default", tf.aliasing_output = 0 : i32}) -> (tensor<2x2xf32>, tensor<5x2xf32>, tensor<5x5xf32>) {
    %0 = tensor.empty() : tensor<2x2xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg0 : tensor<2x5xf32>, tensor<5x2xf32>) outs(%1 : tensor<2x2xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %8 = arith.mulf %in, %in_3 : f32
      %9 = arith.addf %out, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<2x2xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<f32>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<5x2xf32>
    %3 = tensor.empty() : tensor<5x2xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_1 : tensor<5x2xf32>, tensor<5x2xf32>) outs(%3 : tensor<5x2xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %8 = arith.addf %in, %in_3 : f32
      linalg.yield %8 : f32
    } -> tensor<5x2xf32>
    %5 = tensor.empty() : tensor<5x5xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %6 = linalg.fill ins(%cst_2 : f32) outs(%5 : tensor<5x5xf32>) -> tensor<5x5xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<5x2xf32>, tensor<2x5xf32>) outs(%6 : tensor<5x5xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %8 = arith.mulf %in, %in_3 : f32
      %9 = arith.addf %out, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<5x5xf32>
    return %2, %4, %7 : tensor<2x2xf32>, tensor<5x2xf32>, tensor<5x5xf32>
  }
}

// CHECK-NEXT:  module {
// CHECK-NEXT:    memref.global "private" constant @__constant_5x2xf32 : memref<5x2xf32> = dense<1.000000e+00> {alignment = 64 : i64}
// CHECK-NEXT:    memref.global "private" constant @__constant_xf32 : memref<f32> = dense<1.000000e+00> {alignment = 64 : i64}
// CHECK-NEXT:    func.func public @main(%arg0: memref<5x2xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default"}, %arg1: memref<2x5xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default"}, %arg2: memref<5x5xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default", tf.aliasing_output = 2 : i32}, %arg3: memref<2x2xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default", tf.aliasing_output = 0 : i32}) -> (memref<2x2xf32, strided<[?, ?], offset: ?>>, memref<5x2xf32>, memref<5x5xf32, strided<[?, ?], offset: ?>>) {
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      linalg.fill ins(%cst : f32) outs(%arg3 : memref<2x2xf32, strided<[?, ?], offset: ?>>)
// CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg0 : memref<2x5xf32, strided<[?, ?], offset: ?>>, memref<5x2xf32, strided<[?, ?], offset: ?>>) outs(%arg3 : memref<2x2xf32, strided<[?, ?], offset: ?>>) {
// CHECK-NEXT:      ^bb0(%in: f32, %in_1: f32, %out: f32):
// CHECK-NEXT:        %2 = arith.mulf %in, %in_1 : f32
// CHECK-NEXT:        %3 = arith.addf %out, %2 : f32
// CHECK-NEXT:        linalg.yield %3 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %0 = memref.get_global @__constant_xf32 : memref<f32>
// CHECK-NEXT:      %1 = memref.get_global @__constant_5x2xf32 : memref<5x2xf32>
// CHECK-NEXT:      %alloc = memref.alloc() {alignment = 64 : i64} : memref<5x2xf32>
// CHECK-NEXT:      linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1 : memref<5x2xf32, strided<[?, ?], offset: ?>>, memref<5x2xf32>) outs(%alloc : memref<5x2xf32>) {
// CHECK-NEXT:      ^bb0(%in: f32, %in_1: f32, %out: f32):
// CHECK-NEXT:        %2 = arith.addf %in, %in_1 : f32
// CHECK-NEXT:        linalg.yield %2 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      linalg.fill ins(%cst_0 : f32) outs(%arg2 : memref<5x5xf32, strided<[?, ?], offset: ?>>)
// CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<5x2xf32, strided<[?, ?], offset: ?>>, memref<2x5xf32, strided<[?, ?], offset: ?>>) outs(%arg2 : memref<5x5xf32, strided<[?, ?], offset: ?>>) {
// CHECK-NEXT:      ^bb0(%in: f32, %in_1: f32, %out: f32):
// CHECK-NEXT:        %2 = arith.mulf %in, %in_1 : f32
// CHECK-NEXT:        %3 = arith.addf %out, %2 : f32
// CHECK-NEXT:        linalg.yield %3 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.copy %arg3, %arg3 : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-NEXT:      memref.copy %arg2, %arg2 : memref<5x5xf32, strided<[?, ?], offset: ?>> to memref<5x5xf32, strided<[?, ?], offset: ?>>
// CHECK-NEXT:      %cast = memref.cast %alloc : memref<5x2xf32> to memref<5x2xf32, strided<[?, ?], offset: ?>>
// CHECK-NEXT:      return %arg3, %alloc, %arg2 : memref<2x2xf32, strided<[?, ?], offset: ?>>, memref<5x2xf32>, memref<5x5xf32, strided<[?, ?], offset: ?>>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK-NEXT: }