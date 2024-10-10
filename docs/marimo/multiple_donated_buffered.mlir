#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module @jit_multiple_outputs attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  memref.global "private" constant @__constant_5x2xf32 : memref<5x2xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_xf32 : memref<f32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func public @main(%arg0: memref<5x2xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default"}, %arg1: memref<2x5xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default"}, %arg2: memref<5x5xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default", tf.aliasing_output = 2 : i32}, %arg3: memref<2x2xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default", tf.aliasing_output = 0 : i32}) -> (memref<2x2xf32, strided<[?, ?], offset: ?>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, memref<5x2xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, memref<5x5xf32, strided<[?, ?], offset: ?>> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst : f32) outs(%arg3 : memref<2x2xf32, strided<[?, ?], offset: ?>>)
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg0 : memref<2x5xf32, strided<[?, ?], offset: ?>>, memref<5x2xf32, strided<[?, ?], offset: ?>>) outs(%arg3 : memref<2x2xf32, strided<[?, ?], offset: ?>>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in, %in_1 : f32
      %3 = arith.addf %out, %2 : f32
      linalg.yield %3 : f32
    }
    %0 = memref.get_global @__constant_xf32 : memref<f32>
    %1 = memref.get_global @__constant_5x2xf32 : memref<5x2xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<5x2xf32>
    linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1 : memref<5x2xf32, strided<[?, ?], offset: ?>>, memref<5x2xf32>) outs(%alloc : memref<5x2xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.addf %in, %in_1 : f32
      linalg.yield %2 : f32
    }
    %cst_0 = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst_0 : f32) outs(%arg2 : memref<5x5xf32, strided<[?, ?], offset: ?>>)
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<5x2xf32, strided<[?, ?], offset: ?>>, memref<2x5xf32, strided<[?, ?], offset: ?>>) outs(%arg2 : memref<5x5xf32, strided<[?, ?], offset: ?>>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in, %in_1 : f32
      %3 = arith.addf %out, %2 : f32
      linalg.yield %3 : f32
    }
    memref.copy %arg3, %arg3 : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32, strided<[?, ?], offset: ?>>
    memref.copy %arg2, %arg2 : memref<5x5xf32, strided<[?, ?], offset: ?>> to memref<5x5xf32, strided<[?, ?], offset: ?>>
    %cast = memref.cast %alloc : memref<5x2xf32> to memref<5x2xf32, strided<[?, ?], offset: ?>>
    return %arg3, %alloc, %arg2 : memref<2x2xf32, strided<[?, ?], offset: ?>>, memref<5x2xf32>, memref<5x5xf32, strided<[?, ?], offset: ?>>
  }
}
