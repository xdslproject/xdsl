#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @jit_matmul attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: memref<2x3xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default"}, %arg1: memref<3x4xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default"}, %arg2: memref<2x4xf32, strided<[?, ?], offset: ?>> {mhlo.layout_mode = "default", tf.aliasing_output = 0 : i32}) -> (memref<2x4xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4xf32>
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst : f32) outs(%alloc : memref<2x4xf32>)
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<2x3xf32, strided<[?, ?], offset: ?>>, memref<3x4xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<2x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }
    %cast = memref.cast %alloc : memref<2x4xf32> to memref<2x4xf32, strided<[?, ?], offset: ?>>
    return %alloc : memref<2x4xf32>
  }
}
