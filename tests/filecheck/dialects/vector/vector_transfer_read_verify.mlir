// RUN: xdsl-opt --split-input-file --verify-diagnostics %s | filecheck %s

%source, %index, %padding = "test.op"() : () -> (vector<4x3xf32>, index, f32)
"vector.transfer_read"(%source, %index, %index, %padding) <{in_bounds=[true], operandSegmentSizes = array<i32: 1, 3, 1, 0>, permutation_map = affine_map<() -> (0)>}> : (vector<4x3xf32>, index, index, f32) -> vector<1x1x2x3xf32>
// CHECK: operand 'source' at position 0 does not verify:
// CHECK: Unexpected attribute vector<4x3xf32>

// -----

%source, %index, %padding = "test.op"() : () -> (memref<?x?xf32>, index, f32)
"vector.transfer_read"(%source, %index, %index, %index, %padding) <{in_bounds=[true], operandSegmentSizes = array<i32: 1, 3, 1, 0>, permutation_map = affine_map<() -> (0)>}> : (memref<?x?xf32>, index, index, index, f32) -> vector<128xf32>
// CHECK: Expected an index for each memref/tensor dimension

// -----

%source, %index, %padding = "test.op"() : () -> (memref<?x?xf32>, index, f32)
"vector.transfer_read"(%source, %index, %index, %padding) <{in_bounds=[true], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0) -> (d0)>}> : (memref<?x?xf32>, index, index, f32) -> vector<128xf32>
// CHECK: requires a permutation_map with input dims of the same rank as the source type

// -----

%source, %index, %padding = "test.op"() : () -> (memref<?x?xf32>, index, f32)
"vector.transfer_read"(%source, %index, %index, %padding) <{in_bounds=[true, true], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<?x?xf32>, index, index, f32) -> vector<128xf32>
// CHECK: requires a permutation_map with result dims of the same rank as the vector type

// -----

%source, %index, %padding = "test.op"() : () -> (memref<?x?xf32>, index, f32)
"vector.transfer_read"(%source, %index, %index, %padding) <{in_bounds=[true, true], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0 + d1)>}> : (memref<?x?xf32>, index, index, f32) -> vector<128xf32>
// CHECK: requires a projected permutation_map (at most one dim or the zero constant can appear in each result

// -----

%source, %index, %padding = "test.op"() : () -> (memref<?x?xf32>, index, f32)
"vector.transfer_read"(%source, %index, %index, %padding) <{in_bounds=[true], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0 + 1)>}> : (memref<?x?xf32>, index, index, f32) -> vector<128xf32>
// CHECK: requires a projected permutation_map (at most one dim or the zero constant can appear in each result)

// -----

%source, %index, %padding = "test.op"() : () -> (memref<?x?x?xf32>, index, f32)
"vector.transfer_read"(%source, %index, %index, %index, %padding) <{in_bounds=[true, true], operandSegmentSizes = array<i32: 1, 3, 1, 0>, permutation_map = affine_map<(d0, d1, d2) -> (d0, d0)>}> : (memref<?x?x?xf32>, index, index, index, f32) -> vector<3x7xf32>
// CHECK: requires a permutation_map that is a permutation (found one dim used more than once)

// TODO transfer other tests from mlir/test/Dialect/Vector/invalid.mlir once verification for vector element types is implemented
