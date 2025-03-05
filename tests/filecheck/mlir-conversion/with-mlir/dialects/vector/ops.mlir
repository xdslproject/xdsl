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
// See func vector_transfer_ops_0d in mlir/test/Dialect/Vector/ops.mlir

%tensor, %vector, %memref, %f= "test.op"() : () -> (tensor<f32>, vector<f32>, memref<f32>, f32)
// CHECK:      %0, %1, %2, %3 = "test.op"() : () -> (tensor<f32>, vector<f32>, memref<f32>, f32)

%0 = "vector.transfer_read"(%tensor, %f) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 0, 1, 0>, permutation_map =  affine_map<() -> ()>}> : (tensor<f32>, f32) -> vector<f32>
// CHECK-NEXT: %4 = "vector.transfer_read"(%0, %3) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 0, 1, 0>, permutation_map = affine_map<() -> ()>}> : (tensor<f32>, f32) -> vector<f32>

%1 = "vector.transfer_write"(%vector, %tensor) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 0, 0>, permutation_map =  affine_map<() -> ()>}> : (vector<f32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT: %5 = "vector.transfer_write"(%1, %0) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 0, 0>, permutation_map = affine_map<() -> ()>}> : (vector<f32>, tensor<f32>) -> tensor<f32>

%2 = "vector.transfer_read"(%memref, %f) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 0, 1, 0>, permutation_map =  affine_map<() -> ()>}> : (memref<f32>, f32) -> vector<f32>
// CHECK-NEXT: %6 = "vector.transfer_read"(%2, %3) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 0, 1, 0>, permutation_map = affine_map<() -> ()>}> : (memref<f32>, f32) -> vector<f32>

"vector.transfer_write"(%vector, %memref) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 0, 0>, permutation_map =  affine_map<() -> ()>}> : (vector<f32>, memref<f32>) -> ()
// CHECK-NEXT: "vector.transfer_write"(%1, %2) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 0, 0>, permutation_map = affine_map<() -> ()>}> : (vector<f32>, memref<f32>) -> ()

// -----
// Vector transfer ops 0d from higher d
// func vector_transfer_ops_0d_from_higher_d in mlir/test/Dialect/Vector/ops.mlir

%tensor, %memref, %index, %f= "test.op"() : () -> (tensor<?xf32>, memref<?x?xf32>, index, f32)
// CHECK:      %0, %1, %2, %3 = "test.op"() : () -> (tensor<?xf32>, memref<?x?xf32>, index, f32)

%0 = "vector.transfer_read"(%tensor, %index, %f) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 1, 0>, permutation_map = affine_map<(d0) -> ()>}> : (tensor<?xf32>, index, f32) -> vector<f32>
// CHECK-NEXT: %4 = "vector.transfer_read"(%0, %2, %3) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 1, 0>, permutation_map = affine_map<(d0) -> ()>}> : (tensor<?xf32>, index, f32) -> vector<f32>

%1 = "vector.transfer_write"(%0, %tensor, %index) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 1, 0>, permutation_map = affine_map<(d0) -> ()>}> : (vector<f32>, tensor<?xf32>, index) -> tensor<?xf32>
// CHECK-NEXT: %5 = "vector.transfer_write"(%4, %0, %2) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 1, 0>, permutation_map = affine_map<(d0) -> ()>}> : (vector<f32>, tensor<?xf32>, index) -> tensor<?xf32>

%2 = "vector.transfer_read"(%memref, %index, %index, %f) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (memref<?x?xf32>, index, index, f32) -> vector<f32>
// CHECK-NEXT: %6 = "vector.transfer_read"(%1, %2, %2, %3) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (memref<?x?xf32>, index, index, f32) -> vector<f32>

"vector.transfer_write"(%2, %memref, %index, %index) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (vector<f32>, memref<?x?xf32>, index, index) -> ()
// CHECK-NEXT: "vector.transfer_write"(%6, %1, %2, %2) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (vector<f32>, memref<?x?xf32>, index, index) -> ()

// -----
// Vector transfer ops
// func vector_transfer_ops in mlir/test/Dialect/Vector/ops.mlir

%0, %1, %2, %3, %4 = "test.op"() : () -> (memref<?x?xf32>, memref<?x?xvector<4x3xf32>>, memref<?x?xvector<4x3xi32>>, memref<?x?xvector<4x3xindex>>, memref<?x?x?xf32>)
// CHECK:      %0, %1, %2, %3, %4 = "test.op"() : () -> (memref<?x?xf32>, memref<?x?xvector<4x3xf32>>, memref<?x?xvector<4x3xi32>>, memref<?x?xvector<4x3xindex>>, memref<?x?x?xf32>)

%5, %6, %7, %8, %9, %10 = "test.op"() : () -> (index, f32, f32, i32, index, i1)
// CHECK-NEXT: %5, %6, %7, %8, %9, %10 = "test.op"() : () -> (index, f32, f32, i32, index, i1)

%11, %12, %13, %14, %15 = "test.op"() : () -> (vector<4x3xf32>, vector<4x3xi32>, vector<4x3xindex>, vector<5xi1>, vector<4x5xi1>)
// CHECK-NEXT: %11, %12, %13, %14, %15 = "test.op"() : () -> (vector<4x3xf32>, vector<4x3xi32>, vector<4x3xindex>, vector<5xi1>, vector<4x5xi1>)

%16 = "vector.transfer_read"(%0, %5, %5, %7) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0)>}> : (memref<?x?xf32>, index, index, f32) -> vector<128xf32>
// CHECK-NEXT: %16 = "vector.transfer_read"(%0, %5, %5, %7) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0)>}> : (memref<?x?xf32>, index, index, f32) -> vector<128xf32>

%17 = "vector.transfer_read"(%0, %5, %5, %7) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d1, d0)>}> : (memref<?x?xf32>, index, index, f32) -> vector<3x7xf32>
// CHECK-NEXT: %17 = "vector.transfer_read"(%0, %5, %5, %7) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d1, d0)>}> : (memref<?x?xf32>, index, index, f32) -> vector<3x7xf32>

%18 = "vector.transfer_read"(%0, %5, %5, %6) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0)>}> : (memref<?x?xf32>, index, index, f32) -> vector<128xf32>
// CHECK-NEXT: %18 = "vector.transfer_read"(%0, %5, %5, %6) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0)>}> : (memref<?x?xf32>, index, index, f32) -> vector<128xf32>

%19 = "vector.transfer_read"(%0, %5, %5, %6) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d1)>}> : (memref<?x?xf32>, index, index, f32) -> vector<128xf32>
// CHECK-NEXT: %19 = "vector.transfer_read"(%0, %5, %5, %6) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d1)>}> : (memref<?x?xf32>, index, index, f32) -> vector<128xf32>

%20 = "vector.transfer_read"(%1, %5, %5, %11) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<?x?xvector<4x3xf32>>, index, index, vector<4x3xf32>) -> vector<1x1x4x3xf32>
// CHECK-NEXT: %20 = "vector.transfer_read"(%1, %5, %5, %11) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<?x?xvector<4x3xf32>>, index, index, vector<4x3xf32>) -> vector<1x1x4x3xf32>

%21 = "vector.transfer_read"(%1, %5, %5, %11) <{in_bounds = [false, true], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<?x?xvector<4x3xf32>>, index, index, vector<4x3xf32>) -> vector<1x1x4x3xf32>
// CHECK-NEXT: %21 = "vector.transfer_read"(%1, %5, %5, %11) <{in_bounds = [false, true], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<?x?xvector<4x3xf32>>, index, index, vector<4x3xf32>) -> vector<1x1x4x3xf32>

%22 = "vector.transfer_read"(%2, %5, %5, %12) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (memref<?x?xvector<4x3xi32>>, index, index, vector<4x3xi32>) -> vector<5x24xi8>
// CHECK-NEXT: %22 = "vector.transfer_read"(%2, %5, %5, %12) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (memref<?x?xvector<4x3xi32>>, index, index, vector<4x3xi32>) -> vector<5x24xi8>

%23 = "vector.transfer_read"(%3, %5, %5, %13) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (memref<?x?xvector<4x3xindex>>, index, index, vector<4x3xindex>) -> vector<5x48xi8>
// CHECK-NEXT: %23 = "vector.transfer_read"(%3, %5, %5, %13) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (memref<?x?xvector<4x3xindex>>, index, index, vector<4x3xindex>) -> vector<5x48xi8>

%24 = "vector.transfer_read"(%0, %5, %5, %7, %14) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 2, 1, 1>, permutation_map = affine_map<(d0, d1) -> (d1)>}> : (memref<?x?xf32>, index, index, f32, vector<5xi1>) -> vector<5xf32>
// CHECK-NEXT: %24 = "vector.transfer_read"(%0, %5, %5, %7, %14) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 2, 1, 1>, permutation_map = affine_map<(d0, d1) -> (d1)>}> : (memref<?x?xf32>, index, index, f32, vector<5xi1>) -> vector<5xf32>

%25 = "vector.transfer_read"(%4, %5, %5, %5, %7, %15) <{in_bounds = [false, false, true], operandSegmentSizes = array<i32: 1, 3, 1, 1>, permutation_map = affine_map<(d0, d1, d2) -> (d1, d0, 0)>}> : (memref<?x?x?xf32>, index, index, index, f32, vector<4x5xi1>) -> vector<5x4x8xf32>
// CHECK-NEXT: %25 = "vector.transfer_read"(%4, %5, %5, %5, %7, %15) <{in_bounds = [false, false, true], operandSegmentSizes = array<i32: 1, 3, 1, 1>, permutation_map = affine_map<(d0, d1, d2) -> (d1, d0, 0)>}> : (memref<?x?x?xf32>, index, index, index, f32, vector<4x5xi1>) -> vector<5x4x8xf32>

"vector.transfer_write"(%16, %0, %5, %5) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d0)>}> : (vector<128xf32>, memref<?x?xf32>, index, index) -> ()
// CHECK-NEXT: "vector.transfer_write"(%16, %0, %5, %5) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d0)>}> : (vector<128xf32>, memref<?x?xf32>, index, index) -> ()

"vector.transfer_write"(%17, %0, %5, %5) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d1, d0)>}> : (vector<3x7xf32>, memref<?x?xf32>, index, index) -> ()
// CHECK-NEXT: "vector.transfer_write"(%17, %0, %5, %5) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d1, d0)>}> : (vector<3x7xf32>, memref<?x?xf32>, index, index) -> ()

"vector.transfer_write"(%20, %1, %5, %5) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (vector<1x1x4x3xf32>, memref<?x?xvector<4x3xf32>>, index, index) -> ()
// CHECK-NEXT: "vector.transfer_write"(%20, %1, %5, %5) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (vector<1x1x4x3xf32>, memref<?x?xvector<4x3xf32>>, index, index) -> ()

"vector.transfer_write"(%21, %1, %5, %5) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (vector<1x1x4x3xf32>, memref<?x?xvector<4x3xf32>>, index, index) -> ()
// CHECK-NEXT: "vector.transfer_write"(%21, %1, %5, %5) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (vector<1x1x4x3xf32>, memref<?x?xvector<4x3xf32>>, index, index) -> ()

"vector.transfer_write"(%22, %2, %5, %5) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (vector<5x24xi8>, memref<?x?xvector<4x3xi32>>, index, index) -> ()
// CHECK-NEXT: "vector.transfer_write"(%22, %2, %5, %5) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (vector<5x24xi8>, memref<?x?xvector<4x3xi32>>, index, index) -> ()

"vector.transfer_write"(%23, %3, %5, %5) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (vector<5x48xi8>, memref<?x?xvector<4x3xindex>>, index, index) -> ()
// CHECK-NEXT: "vector.transfer_write"(%23, %3, %5, %5) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (vector<5x48xi8>, memref<?x?xvector<4x3xindex>>, index, index) -> ()

"vector.transfer_write"(%24, %0, %5, %5, %14) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 2, 1>, permutation_map = affine_map<(d0, d1) -> (d1)>}> : (vector<5xf32>, memref<?x?xf32>, index, index, vector<5xi1>) -> ()
// CHECK-NEXT: "vector.transfer_write"(%24, %0, %5, %5, %14) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 2, 1>, permutation_map = affine_map<(d0, d1) -> (d1)>}> : (vector<5xf32>, memref<?x?xf32>, index, index, vector<5xi1>) -> ()

// -----
// Vector transfer ops tensor
// func vector_transfer_ops_tensor in mlir/test/Dialect/Vector/ops.mlir

%0, %1, %2, %3 = "test.op"() : () -> (tensor<?x?xf32>, tensor<?x?xvector<4x3xf32>>, tensor<?x?xvector<4x3xi32>>, tensor<?x?xvector<4x3xindex>>)
// CHECK:      %0, %1, %2, %3 = "test.op"() : () -> (tensor<?x?xf32>, tensor<?x?xvector<4x3xf32>>, tensor<?x?xvector<4x3xi32>>, tensor<?x?xvector<4x3xindex>>)

%4, %5, %6, %7, %8 = "test.op"() : () -> (index, f32, f32, i32, index)
// CHECK-NEXT: %4, %5, %6, %7, %8 = "test.op"() : () -> (index, f32, f32, i32, index)

%9, %10, %11 = "test.op"() : () -> (vector<4x3xf32>, vector<4x3xi32>, vector<4x3xindex>)
// CHECK-NEXT: %9, %10, %11 = "test.op"() : () -> (vector<4x3xf32>, vector<4x3xi32>, vector<4x3xindex>)

%12 = "vector.transfer_read"(%0, %4, %4, %6) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0)>}> : (tensor<?x?xf32>, index, index, f32) -> vector<128xf32>
// CHECK-NEXT: %12 = "vector.transfer_read"(%0, %4, %4, %6) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0)>}> : (tensor<?x?xf32>, index, index, f32) -> vector<128xf32>

%13 = "vector.transfer_read"(%0, %4, %4, %6) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d1, d0)>}> : (tensor<?x?xf32>, index, index, f32) -> vector<3x7xf32>
// CHECK-NEXT: %13 = "vector.transfer_read"(%0, %4, %4, %6) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d1, d0)>}> : (tensor<?x?xf32>, index, index, f32) -> vector<3x7xf32>

%14 = "vector.transfer_read"(%0, %4, %4, %5) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0)>}> : (tensor<?x?xf32>, index, index, f32) -> vector<128xf32>
// CHECK-NEXT: %14 = "vector.transfer_read"(%0, %4, %4, %5) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0)>}> : (tensor<?x?xf32>, index, index, f32) -> vector<128xf32>

%15 = "vector.transfer_read"(%0, %4, %4, %5) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d1)>}> : (tensor<?x?xf32>, index, index, f32) -> vector<128xf32>
// CHECK-NEXT: %15 = "vector.transfer_read"(%0, %4, %4, %5) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d1)>}> : (tensor<?x?xf32>, index, index, f32) -> vector<128xf32>

%16 = "vector.transfer_read"(%1, %4, %4, %9) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (tensor<?x?xvector<4x3xf32>>, index, index, vector<4x3xf32>) -> vector<1x1x4x3xf32>
// CHECK-NEXT: %16 = "vector.transfer_read"(%1, %4, %4, %9) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (tensor<?x?xvector<4x3xf32>>, index, index, vector<4x3xf32>) -> vector<1x1x4x3xf32>

%17 = "vector.transfer_read"(%1, %4, %4, %9) <{in_bounds = [false, true], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (tensor<?x?xvector<4x3xf32>>, index, index, vector<4x3xf32>) -> vector<1x1x4x3xf32>
// CHECK-NEXT: %17 = "vector.transfer_read"(%1, %4, %4, %9) <{in_bounds = [false, true], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (tensor<?x?xvector<4x3xf32>>, index, index, vector<4x3xf32>) -> vector<1x1x4x3xf32>

%18 = "vector.transfer_read"(%2, %4, %4, %10) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (tensor<?x?xvector<4x3xi32>>, index, index, vector<4x3xi32>) -> vector<5x24xi8>
// CHECK-NEXT: %18 = "vector.transfer_read"(%2, %4, %4, %10) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (tensor<?x?xvector<4x3xi32>>, index, index, vector<4x3xi32>) -> vector<5x24xi8>

%19 = "vector.transfer_read"(%3, %4, %4, %11) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (tensor<?x?xvector<4x3xindex>>, index, index, vector<4x3xindex>) -> vector<5x48xi8>
// CHECK-NEXT: %19 = "vector.transfer_read"(%3, %4, %4, %11) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (tensor<?x?xvector<4x3xindex>>, index, index, vector<4x3xindex>) -> vector<5x48xi8>

%20 = "vector.transfer_write"(%12, %0, %4, %4) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d0)>}> : (vector<128xf32>, tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
// CHECK-NEXT: %20 = "vector.transfer_write"(%12, %0, %4, %4) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d0)>}> : (vector<128xf32>, tensor<?x?xf32>, index, index) -> tensor<?x?xf32>

%21 = "vector.transfer_write"(%13, %0, %4, %4) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d1, d0)>}> : (vector<3x7xf32>, tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
// CHECK-NEXT: %21 = "vector.transfer_write"(%13, %0, %4, %4) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d1, d0)>}> : (vector<3x7xf32>, tensor<?x?xf32>, index, index) -> tensor<?x?xf32>

%22 = "vector.transfer_write"(%16, %1, %4, %4) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>, index, index) -> tensor<?x?xvector<4x3xf32>>
// CHECK-NEXT: %22 = "vector.transfer_write"(%16, %1, %4, %4) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>, index, index) -> tensor<?x?xvector<4x3xf32>>

%23 = "vector.transfer_write"(%17, %1, %4, %4) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>, index, index) -> tensor<?x?xvector<4x3xf32>>
// CHECK-NEXT: %23 = "vector.transfer_write"(%17, %1, %4, %4) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d0, d1)>}> : (vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>, index, index) -> tensor<?x?xvector<4x3xf32>>

%24 = "vector.transfer_write"(%18, %2, %4, %4) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (vector<5x24xi8>, tensor<?x?xvector<4x3xi32>>, index, index) -> tensor<?x?xvector<4x3xi32>>
// CHECK-NEXT: %24 = "vector.transfer_write"(%18, %2, %4, %4) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (vector<5x24xi8>, tensor<?x?xvector<4x3xi32>>, index, index) -> tensor<?x?xvector<4x3xi32>>

%25 = "vector.transfer_write"(%19, %3, %4, %4) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (vector<5x48xi8>, tensor<?x?xvector<4x3xindex>>, index, index) -> tensor<?x?xvector<4x3xindex>>
// CHECK-NEXT: %25 = "vector.transfer_write"(%19, %3, %4, %4) <{in_bounds = [], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> ()>}> : (vector<5x48xi8>, tensor<?x?xvector<4x3xindex>>, index, index) -> tensor<?x?xvector<4x3xindex>>
