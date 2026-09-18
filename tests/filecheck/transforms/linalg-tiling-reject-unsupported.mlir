// RUN: xdsl-opt -p test-linalg-tiling --split-input-file --verify-diagnostics %s | filecheck %s

builtin.module {
  %input = "test.op"() : () -> memref<4x4xf32>
  %output = "test.op"() : () -> memref<4x4xf32>
  linalg.generic {
      indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel"]
  } ins(%input : memref<4x4xf32>) outs(%output : memref<4x4xf32>) attrs = {test_tile_sizes = array<i32: -2, 2>} {
  ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
}
// CHECK: negative tile sizes are not supported

// -----

builtin.module {
  %input = "test.op"() : () -> memref<4x4xf32, affine_map<(d0, d1) -> (d0 * 4 + d1)>>
  %output = "test.op"() : () -> memref<4x4xf32, affine_map<(d0, d1) -> (d0 * 4 + d1)>>
  linalg.generic {
      indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel"]
  } ins(%input : memref<4x4xf32, affine_map<(d0, d1) -> (d0 * 4 + d1)>>) outs(%output : memref<4x4xf32, affine_map<(d0, d1) -> (d0 * 4 + d1)>>) attrs = {test_tile_sizes = array<i32: 2, 2>} {
  ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
}
// CHECK: cannot infer memref.subview result type from non-strided source type
