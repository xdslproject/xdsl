#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  memref.global "private" constant @__constant_5x5xf32 : memref<5x5xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_xf32 : memref<f32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func public @main(%arg0: memref<5x5xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default"}, %arg1: memref<5x5xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default"}, %arg2: memref<5x5xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default", tf.aliasing_output = 0 : i32}) -> memref<5x5xf32, strided<[?, ?], offset: ?>> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<5x5xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<5x5xf32, strided<[?, ?], offset: ?>>, memref<5x5xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<5x5xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    }
    %0 = memref.get_global @__constant_xf32 : memref<f32>
    %1 = memref.get_global @__constant_5x5xf32 : memref<5x5xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc, %1 : memref<5x5xf32>, memref<5x5xf32>) outs(%arg2 : memref<5x5xf32, strided<[?, ?], offset: ?>>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    }
    memref.copy %arg2, %arg2 : memref<5x5xf32, strided<[?, ?], offset: ?>> to memref<5x5xf32, strided<[?, ?], offset: ?>>
    return %arg2 : memref<5x5xf32, strided<[?, ?], offset: ?>>
  }
}
