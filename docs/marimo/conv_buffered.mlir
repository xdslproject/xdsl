#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10) -> (d10, d0, d2 + d3, d4 + d5, d6 + d7, d8 + d9)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10) -> (d1, d0, d3, d5, d7, d9)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10) -> (d10, d1, d2, d4, d6, d8)>
#map5 = affine_map<(d0, d1, d2, d3) -> (-d0, -d1, -d2 + 2, -d3 + 2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func public @main(%arg0: memref<1x1x10x10xf32, strided<[?, ?, ?, ?], offset: ?>> {mhlo.layout_mode = "default"}, %arg1: memref<1x1x3x3xf32, strided<[?, ?, ?, ?], offset: ?>> {mhlo.layout_mode = "default"}, %arg2: memref<1x1x8x8xf32, strided<[?, ?, ?, ?], offset: ?>> {mhlo.layout_mode = "default", tf.aliasing_output = 0 : i32}) -> memref<1x1x8x8xf32> {
    %0 = call @_flip(%arg1) : (memref<1x1x3x3xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x1x3x3xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x1x10x10xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x1x10x10xf32, strided<[?, ?, ?, ?], offset: ?>>) outs(%alloc : memref<1x1x1x1x10x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x1x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%0 : memref<1x1x3x3xf32>) outs(%alloc_0 : memref<1x1x1x1x3x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x1x8x8xf32>
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<1x1x1x1x8x8xf32>)
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "reduction", "parallel", "reduction", "parallel", "reduction", "parallel"]} ins(%alloc, %alloc_0 : memref<1x1x1x1x10x10xf32>, memref<1x1x1x1x3x3xf32>) outs(%alloc_1 : memref<1x1x1x1x8x8xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %1 = arith.mulf %in, %in_2 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
    %collapse_shape = memref.collapse_shape %alloc_1 [[0], [1], [2, 3, 4], [5]] : memref<1x1x1x1x8x8xf32> into memref<1x1x8x8xf32>
    %cast = memref.cast %collapse_shape : memref<1x1x8x8xf32> to memref<1x1x8x8xf32, strided<[?, ?, ?, ?], offset: ?>>
    return %collapse_shape : memref<1x1x8x8xf32>
  }
  func.func private @_flip(%arg0: memref<1x1x3x3xf32, strided<[?, ?, ?, ?], offset: ?>> {mhlo.layout_mode = "default"}) -> memref<1x1x3x3xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x3x3xf32>
    linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x1x3x3xf32, strided<[?, ?, ?, ?], offset: ?>>) outs(%alloc : memref<1x1x3x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %cast = memref.cast %alloc : memref<1x1x3x3xf32> to memref<1x1x3x3xf32, strided<[?, ?, ?, ?], offset: ?>>
    return %alloc : memref<1x1x3x3xf32>
  }
}
