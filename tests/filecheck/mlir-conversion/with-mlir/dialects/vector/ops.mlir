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

%1 = "vector.transfer_read"(%tensor, %f) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, permutation_map =  affine_map<() -> ()>}> : (tensor<f32>, f32) -> vector<f32>
// CHECK-NEXT: %4 = "vector.transfer_read"(%0, %3) <{"operandSegmentSizes" = array<i32: 1, 0, 1, 0>, "permutation_map" = affine_map<() -> ()>}> : (tensor<f32>, f32) -> vector<f32>

%2 = "vector.transfer_write"(%vector, %tensor) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, permutation_map =  affine_map<() -> ()>}> : (vector<f32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT: %5 = "vector.transfer_write"(%1, %0) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 0>, "permutation_map" = affine_map<() -> ()>}> : (vector<f32>, tensor<f32>) -> tensor<f32>

%3 = "vector.transfer_read"(%memref, %f) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, permutation_map =  affine_map<() -> ()>}> : (memref<f32>, f32) -> vector<f32>
// CHECK-NEXT: %6 = "vector.transfer_read"(%2, %3) <{"operandSegmentSizes" = array<i32: 1, 0, 1, 0>, "permutation_map" = affine_map<() -> ()>}> : (memref<f32>, f32) -> vector<f32>

"vector.transfer_write"(%vector, %memref) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, permutation_map =  affine_map<() -> ()>}> : (vector<f32>, memref<f32>) -> ()
// CHECK-NEXT: "vector.transfer_write"(%1, %2) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 0>, "permutation_map" = affine_map<() -> ()>}> : (vector<f32>, memref<f32>) -> ()
