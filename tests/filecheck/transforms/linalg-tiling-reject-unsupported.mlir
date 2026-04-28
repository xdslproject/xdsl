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
  %output = "test.op"() : () -> tensor<4x4xf32>
  %result = linalg.generic {
      indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<4x4xf32>) outs(%output : tensor<4x4xf32>) attrs = {test_tile_sizes = array<i32: 2, 2>} {
  ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<4x4xf32>
  "test.op"(%result) : (tensor<4x4xf32>) -> ()
}
// CHECK: tiling linalg.generic with tensor results is not supported yet

// -----

builtin.module {
  %input = "test.op"() : () -> memref<4x4xf32>
  %output = "test.op"() : () -> memref<4x4xf32>
  linalg.generic {
      indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel"]
  } ins(%input : memref<4x4xf32>) outs(%output : memref<4x4xf32>) attrs = {test_tile_sizes = array<i32: 2, 2>} {
  ^bb0(%in: f32, %out: f32):
      %i = linalg.index 0 : index
      %unused = "test.op"(%i) : (index) -> f32
      linalg.yield %unused : f32
  }
}
// CHECK: tiling linalg.generic using linalg.index is not supported yet

// -----

builtin.module {
  %input = "test.op"() : () -> memref<4x4xf32>
  %output = "test.op"() : () -> memref<4x4xf32>
  linalg.generic {
      indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i, j)>
      ],
      iterator_types = ["parallel", "reduction"]
  } ins(%input : memref<4x4xf32>) outs(%output : memref<4x4xf32>) attrs = {test_tile_sizes = array<i32: 2, 2>} {
  ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
}
// CHECK: tiling of non-parallel iterator dimensions is not supported yet

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
// CHECK: tiling linalg.generic with non-memref operands is not supported yet

// -----

builtin.module {
  %input = "test.op"() : () -> memref<?x4xf32>
  %output = "test.op"() : () -> memref<?x4xf32>
  linalg.generic {
      indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel"]
  } ins(%input : memref<?x4xf32>) outs(%output : memref<?x4xf32>) attrs = {test_tile_sizes = array<i32: 2, 2>} {
  ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
}
// CHECK: tiling linalg.generic with dynamic operand shapes is not supported yet

// -----

builtin.module {
  %input = "test.op"() : () -> memref<4x4xf32>
  %output = "test.op"() : () -> memref<4x4xf32>
  linalg.generic {
      indexing_maps = [
          affine_map<(i, j) -> (i + j)>,
          affine_map<(i, j) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel"]
  } ins(%input : memref<4x4xf32>) outs(%output : memref<4x4xf32>) attrs = {test_tile_sizes = array<i32: 2, 2>} {
  ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
}
// CHECK: tiling linalg.generic with non-projected-permutation indexing maps is not supported yet

// -----

builtin.module {
  %input = "test.op"() : () -> memref<5x4xf32>
  %output = "test.op"() : () -> memref<5x4xf32>
  linalg.generic {
      indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel"]
  } ins(%input : memref<5x4xf32>) outs(%output : memref<5x4xf32>) attrs = {test_tile_sizes = array<i32: 2, 2>} {
  ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
}
// CHECK: partial tiles are not supported yet

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
// CHECK: tiling memrefs with layout affine_map<{{.*}}> is not supported yet

// -----

builtin.module {
  %input = "test.op"() : () -> memref<4x4xf32, strided<[?, 1]>>
  %output = "test.op"() : () -> memref<4x4xf32, strided<[?, 1]>>
  linalg.generic {
      indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel"]
  } ins(%input : memref<4x4xf32, strided<[?, 1]>>) outs(%output : memref<4x4xf32, strided<[?, 1]>>) attrs = {test_tile_sizes = array<i32: 2, 2>} {
  ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
}
// CHECK: tiling memrefs with dynamic strides is not supported yet
