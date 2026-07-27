// RUN: xdsl-opt --split-input-file --verify-diagnostics %s | filecheck %s

%source, %index = "test.op"() : () -> (vector<4x3xf32>, index)
"vector.transfer_write"(%source, %source, %index, %index) <{in_bounds=[true], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<() -> (0)>}> : (vector<4x3xf32>, vector<4x3xf32>, index, index) -> ()
// CHECK: operand 'source' at position 1 does not verify:
// CHECK: Unexpected attribute vector<4x3xf32>

// -----

%source, %vector, %index = "test.op"() : () -> (memref<?x?xf32>, vector<128xf32>, index)
"vector.transfer_write"(%vector, %source, %index, %index, %index) <{in_bounds=[true], operandSegmentSizes = array<i32: 1, 1, 3, 0>, permutation_map = affine_map<() -> (0)>}> : (vector<128xf32>, memref<?x?xf32>, index, index, index) -> ()
// CHECK: Expected an index for each memref/tensor dimension

// -----

%source, %vector, %index = "test.op"() : () -> (memref<?x?xf32>, vector<128xf32>, index)
"vector.transfer_write"(%vector, %source, %index, %index) <{in_bounds=[true], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0) -> (d0)>}> : (vector<128xf32>, memref<?x?xf32>, index, index) -> ()
// CHECK: requires a permutation_map with input dims of the same rank as the source type

// -----

%source, %vector, %index = "test.op"() : () -> (memref<?x?xf32>, vector<128xf32>, index)
"vector.transfer_write"(%vector, %source, %index, %index) <{in_bounds=[true, true], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (vector<128xf32>, memref<?x?xf32>, index, index) -> ()
// CHECK: requires a permutation_map with result dims of the same rank as the vector type

// -----

%source, %vector, %index = "test.op"() : () -> (memref<?x?xf32>, vector<128xf32>, index)
"vector.transfer_write"(%vector, %source, %index, %index) <{in_bounds=[true, true], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d0 + d1)>}> : (vector<128xf32>, memref<?x?xf32>, index, index) -> ()
// CHECK: requires a projected permutation_map (at most one dim or the zero constant can appear in each result)

// -----

%source, %vector, %index = "test.op"() : () -> (memref<?x?xf32>, vector<128xf32>, index)
"vector.transfer_write"(%vector, %source, %index, %index) <{in_bounds=[true], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d0 + 1)>}> : (vector<128xf32>, memref<?x?xf32>, index, index) -> ()
// CHECK: requires a projected permutation_map (at most one dim or the zero constant can appear in each result)

// -----

%source, %vector, %index = "test.op"() : () -> (memref<?x?x?xf32>, vector<3x7xf32>, index)
"vector.transfer_write"(%vector, %source, %index, %index, %index) <{in_bounds=[true, true], operandSegmentSizes = array<i32: 1, 1, 3, 0>, permutation_map = affine_map<(d0, d1, d2) -> (d0, d0)>}> : (vector<3x7xf32>, memref<?x?x?xf32>, index, index, index) -> ()
// CHECK: requires a permutation_map that is a permutation (found one dim used more than once)

// -----

%source, %vector, %index = "test.op"() : () -> (memref<?xf32>, vector<7xf32>, index)
"vector.transfer_write"(%vector, %source, %index) <{in_bounds=[true], operandSegmentSizes = array<i32: 1, 1, 1, 0>, permutation_map = affine_map<(d0) -> (0)>}> : (vector<7xf32>, memref<?xf32>, index) -> ()
// CHECK: should not have broadcast dimensions

// TODO transfer other tests from mlir/test/Dialect/Vector/invalid.mlir once verification for vector element types is implemented
