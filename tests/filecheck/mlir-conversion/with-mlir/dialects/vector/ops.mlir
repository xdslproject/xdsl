// RUN: xdsl-opt --split-input-file --print-op-generic %s | mlir-opt --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s


%vector0, %vector1, %i0= "test.op"() : () -> (vector<index>, vector<3xindex>, index)
// CHECK:      %0, %1, %2 = "test.op"() : () -> (vector<index>, vector<3xindex>, index)

%0 = "vector.insertelement"(%i0, %vector0) : (index, vector<index>) -> vector<index>
// CHECK-NEXT: %3 = "vector.insertelement"(%2, %0) : (index, vector<index>) -> vector<index>

%1 = "vector.insertelement"(%i0, %vector1, %i0) : (index, vector<3xindex>, index) -> vector<3xindex>
// CHECK-NEXT: %4 = "vector.insertelement"(%2, %1, %2) : (index, vector<3xindex>, index) -> vector<3xindex>

%2 = "vector.extractelement"(%vector1, %i0) : (vector<3xindex>, index) -> index
// CHECK-NEXT: %5 = "vector.extractelement"(%1, %2) : (vector<3xindex>, index) -> index

%3 = "vector.extractelement"(%vector0) : (vector<index>) -> index
// CHECK-NEXT: %6 = "vector.extractelement"(%0) : (vector<index>) -> index


// -----
// Vector transfer ops 0d

%tensor, %vector, %memref, %f= "test.op"() : () -> (tensor<f32>, vector<f32>, memref<f32>, f32)
// CHECK:      %0, %1, %2, %3 = "test.op"() : () -> (tensor<f32>, vector<f32>, memref<f32>, f32)

%0 = "vector.transfer_read"(%tensor, %f) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, permutation_map =  affine_map<() -> ()>}> : (tensor<f32>, f32) -> vector<f32>
// CHECK-NEXT: %4 = "vector.transfer_read"(%0, %3) <{"operandSegmentSizes" = array<i32: 1, 0, 1, 0>, "permutation_map" = affine_map<() -> ()>}> : (tensor<f32>, f32) -> vector<f32>

%1 = "vector.transfer_write"(%vector, %tensor) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, permutation_map =  affine_map<() -> ()>}> : (vector<f32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT: %5 = "vector.transfer_write"(%1, %0) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 0>, "permutation_map" = affine_map<() -> ()>}> : (vector<f32>, tensor<f32>) -> tensor<f32>

%2 = "vector.transfer_read"(%memref, %f) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, permutation_map =  affine_map<() -> ()>}> : (memref<f32>, f32) -> vector<f32>
// CHECK-NEXT: %6 = "vector.transfer_read"(%2, %3) <{"operandSegmentSizes" = array<i32: 1, 0, 1, 0>, "permutation_map" = affine_map<() -> ()>}> : (memref<f32>, f32) -> vector<f32>

"vector.transfer_write"(%vector, %memref) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, permutation_map =  affine_map<() -> ()>}> : (vector<f32>, memref<f32>) -> ()
// CHECK-NEXT: "vector.transfer_write"(%1, %2) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 0>, "permutation_map" = affine_map<() -> ()>}> : (vector<f32>, memref<f32>) -> ()

// -----
// Vector transfer ops 0d from higher d

%tensor, %memref, %index, %f= "test.op"() : () -> (tensor<?xf32>, memref<?x?xf32>, index, f32)
// CHECK:      %0, %1, %2, %3 = "test.op"() : () -> (tensor<?xf32>, memref<?x?xf32>, index, f32)

%0 = "vector.transfer_read"(%tensor, %index, %f) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, permutation_map = affine_map<(d0) -> ()>}> : (tensor<?xf32>, index, f32) -> vector<f32>
// CHECK-NEXT: %4 = "vector.transfer_read"(%0, %2, %3) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>, "permutation_map" = affine_map<(d0) -> ()>}> : (tensor<?xf32>, index, f32) -> vector<f32>

%1 = "vector.transfer_write"(%0, %tensor, %index) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, permutation_map = affine_map<(d0) -> ()>}> : (vector<f32>, tensor<?xf32>, index) -> tensor<?xf32>
// CHECK-NEXT: %5 = "vector.transfer_write"(%4, %0, %2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>, "permutation_map" = affine_map<(d0) -> ()>}> : (vector<f32>, tensor<?xf32>, index) -> tensor<?xf32>

%2 = "vector.transfer_read"(%memref, %index, %index, %f) <{operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (memref<?x?xf32>, index, index, f32) -> vector<f32>
// CHECK-NEXT: %6 = "vector.transfer_read"(%1, %2, %2, %3) <{"operandSegmentSizes" = array<i32: 1, 2, 1, 0>, "permutation_map" = affine_map<(d0, d1) -> ()>}> : (memref<?x?xf32>, index, index, f32) -> vector<f32>

"vector.transfer_write"(%2, %memref, %index, %index) <{operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (vector<f32>, memref<?x?xf32>, index, index) -> ()
// CHECK-NEXT: "vector.transfer_write"(%6, %1, %2, %2) <{"operandSegmentSizes" = array<i32: 1, 1, 2, 0>, "permutation_map" = affine_map<(d0, d1) -> ()>}> : (vector<f32>, memref<?x?xf32>, index, index) -> ()
