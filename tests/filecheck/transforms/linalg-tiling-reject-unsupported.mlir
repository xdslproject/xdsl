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
  %input = "test.op"() : () -> tensor<4x4xf32>
  %out_tensor = "test.op"() : () -> tensor<4x4xf32>
  %out_memref = "test.op"() : () -> memref<4x4xf32>
  %result = "linalg.generic"(%input, %out_tensor, %out_memref) <{
      indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
      operandSegmentSizes = array<i32: 1, 2>
  }> ({
  ^bb0(%in: f32, %a: f32, %b: f32):
      "linalg.yield"(%in) : (f32) -> ()
  }) {test_tile_sizes = array<i32: 2, 2>} : (tensor<4x4xf32>, tensor<4x4xf32>, memref<4x4xf32>) -> tensor<4x4xf32>
  "test.op"(%result) : (tensor<4x4xf32>) -> ()
}
// CHECK: tiling a linalg op with a mix of memref and tensor operands is not supported

// -----

builtin.module {
  %input = "test.op"() : () -> tensor<4x4xf32>
  %output = "test.op"() : () -> memref<4x4xf32>
  linalg.generic {
      indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<4x4xf32>) outs(%output : memref<4x4xf32>) attrs = {test_tile_sizes = array<i32: 2, 2>} {
  ^bb0(%in: f32, %out: f32):
      linalg.yield %out : f32
  }
}
// CHECK: tiling a linalg op with a mix of memref and tensor operands is not supported

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
