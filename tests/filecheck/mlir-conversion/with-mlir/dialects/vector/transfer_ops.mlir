// RUN: xdsl-opt --print-op-generic --split-input-file %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect --split-input-file | xdsl-opt --split-input-file | filecheck %s
// RUN: xdsl-opt --split-input-file %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect --split-input-file | xdsl-opt --split-input-file | filecheck %s

// Vector transfer ops 0d
// See func vector_transfer_ops_0d in mlir/test/Dialect/Vector/ops.mlir

%tensor, %vector, %memref, %f= "test.op"() : () -> (tensor<f32>, vector<f32>, memref<f32>, f32)
// CHECK:      %0, %1, %2, %3 = "test.op"() : () -> (tensor<f32>, vector<f32>, memref<f32>, f32)

%0 = vector.transfer_read %tensor[], %f : tensor<f32>, vector<f32>
// CHECK-NEXT: %4 = vector.transfer_read %0[], %3 : tensor<f32>, vector<f32>

%1 = vector.transfer_write %vector, %tensor [] : vector<f32>, tensor<f32>
// CHECK-NEXT: %5 = vector.transfer_write %1, %0[] : vector<f32>, tensor<f32>

%2 = vector.transfer_read %memref[], %f : memref<f32>, vector<f32>
// CHECK-NEXT: %6 = vector.transfer_read %2[], %3 : memref<f32>, vector<f32>

vector.transfer_write %vector, %memref [] : vector<f32>, memref<f32>
// CHECK-NEXT: vector.transfer_write %1, %2[] : vector<f32>, memref<f32>

// -----
// Vector transfer ops 0d from higher d
// func vector_transfer_ops_0d_from_higher_d in mlir/test/Dialect/Vector/ops.mlir

%tensor, %memref, %index, %f= "test.op"() : () -> (tensor<?xf32>, memref<?x?xf32>, index, f32)
// CHECK:      %0, %1, %2, %3 = "test.op"() : () -> (tensor<?xf32>, memref<?x?xf32>, index, f32)

%0 = vector.transfer_read %tensor[%index], %f : tensor<?xf32>, vector<f32>
// CHECK-NEXT: %4 = vector.transfer_read %0[%2], %3 : tensor<?xf32>, vector<f32>

%1 = vector.transfer_write %0, %tensor [%index] : vector<f32>, tensor<?xf32>
// CHECK-NEXT: %5 = vector.transfer_write %4, %0[%2] : vector<f32>, tensor<?xf32>

%2 = vector.transfer_read %memref[%index, %index], %f : memref<?x?xf32>, vector<f32>
// CHECK-NEXT: %6 = vector.transfer_read %1[%2, %2], %3 : memref<?x?xf32>, vector<f32>

vector.transfer_write %2, %memref [%index, %index] : vector<f32>, memref<?x?xf32>
// CHECK-NEXT: vector.transfer_write %6, %1[%2, %2] : vector<f32>, memref<?x?xf32>

// -----
// Vector transfer ops
// func vector_transfer_ops in mlir/test/Dialect/Vector/ops.mlir

%0, %1, %2, %3, %4 = "test.op"() : () -> (memref<?x?xf32>, memref<?x?xvector<4x3xf32>>, memref<?x?xvector<4x3xi32>>, memref<?x?xvector<4x3xindex>>, memref<?x?x?xf32>)
// CHECK:      %0, %1, %2, %3, %4 = "test.op"() : () -> (memref<?x?xf32>, memref<?x?xvector<4x3xf32>>, memref<?x?xvector<4x3xi32>>, memref<?x?xvector<4x3xindex>>, memref<?x?x?xf32>)

%5, %6, %7, %8, %9, %10 = "test.op"() : () -> (index, f32, f32, i32, index, i1)
// CHECK-NEXT: %5, %6, %7, %8, %9, %10 = "test.op"() : () -> (index, f32, f32, i32, index, i1)

%11, %12, %13, %14, %15 = "test.op"() : () -> (vector<4x3xf32>, vector<4x3xi32>, vector<4x3xindex>, vector<5xi1>, vector<4x5xi1>)
// CHECK-NEXT: %11, %12, %13, %14, %15 = "test.op"() : () -> (vector<4x3xf32>, vector<4x3xi32>, vector<4x3xindex>, vector<5xi1>, vector<4x5xi1>)

%16 = vector.transfer_read %0[%5, %5], %7 {permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<?x?xf32>, vector<128xf32>
// CHECK-NEXT: %16 = vector.transfer_read %0[%5, %5], %7 {permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<?x?xf32>, vector<128xf32>

%17 = vector.transfer_read %0[%5, %5], %7 {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<?x?xf32>, vector<3x7xf32>
// CHECK-NEXT: %17 = vector.transfer_read %0[%5, %5], %7 {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<?x?xf32>, vector<3x7xf32>

%18 = vector.transfer_read %0[%5, %5], %6 {permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<?x?xf32>, vector<128xf32>
// CHECK-NEXT: %18 = vector.transfer_read %0[%5, %5], %6 {permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<?x?xf32>, vector<128xf32>

%19 = vector.transfer_read %0[%5, %5], %6 : memref<?x?xf32>, vector<128xf32>
// CHECK-NEXT: %19 = vector.transfer_read %0[%5, %5], %6 : memref<?x?xf32>, vector<128xf32>

%20 = vector.transfer_read %1[%5, %5], %11 : memref<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
// CHECK-NEXT: %20 = vector.transfer_read %1[%5, %5], %11 : memref<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>

%21 = vector.transfer_read %1[%5, %5], %11 {in_bounds = [false, true]} : memref<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
// CHECK-NEXT: %21 = vector.transfer_read %1[%5, %5], %11 {in_bounds = [false, true]} : memref<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>

%22 = vector.transfer_read %2[%5, %5], %12 : memref<?x?xvector<4x3xi32>>, vector<5x24xi8>
// CHECK-NEXT: %22 = vector.transfer_read %2[%5, %5], %12 : memref<?x?xvector<4x3xi32>>, vector<5x24xi8>

%23 = vector.transfer_read %3[%5, %5], %13 : memref<?x?xvector<4x3xindex>>, vector<5x48xi8>
// CHECK-NEXT: %23 = vector.transfer_read %3[%5, %5], %13 : memref<?x?xvector<4x3xindex>>, vector<5x48xi8>

%24 = vector.transfer_read %0[%5, %5], %7, %14 : memref<?x?xf32>, vector<5xf32>
// CHECK-NEXT: %24 = vector.transfer_read %0[%5, %5], %7, %14 : memref<?x?xf32>, vector<5xf32>

%25 = vector.transfer_read %4[%5, %5, %5], %7, %15 {in_bounds = [false, false, true], permutation_map = affine_map
<(d0, d1, d2) -> (d1, d0, 0)>} : memref<?x?x?xf32>, vector<5x4x8xf32>
// CHECK-NEXT: %25 = vector.transfer_read %4[%5, %5, %5], %7, %15 {in_bounds = [false, false, true], permutation_map = affine_map<(d0, d1, d2) -> (d1, d0, 0)>} : memref<?x?x?xf32>, vector<5x4x8xf32>

vector.transfer_write %16, %0[%5, %5] {permutation_map = affine_map<(d0, d1) -> (d0)>} : vector<128xf32>, memref<?x?xf32>
// CHECK-NEXT: vector.transfer_write %16, %0[%5, %5] {permutation_map = affine_map<(d0, d1) -> (d0)>} : vector<128xf32>, memref<?x?xf32>

vector.transfer_write %17, %0[%5, %5] {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : vector<3x7xf32>, memref<?x?xf32>
// CHECK-NEXT: vector.transfer_write %17, %0[%5, %5] {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : vector<3x7xf32>, memref<?x?xf32>

vector.transfer_write %20, %1[%5, %5] {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} : vector<1x1x4x3xf32>, memref<?x?xvector<4x3xf32>>
// CHECK-NEXT: vector.transfer_write %20, %1[%5, %5] : vector<1x1x4x3xf32>, memref<?x?xvector<4x3xf32>>

vector.transfer_write %21, %1[%5, %5] {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} : vector<1x1x4x3xf32>, memref<?x?xvector<4x3xf32>>
// CHECK-NEXT: vector.transfer_write %21, %1[%5, %5] : vector<1x1x4x3xf32>, memref<?x?xvector<4x3xf32>>

vector.transfer_write %22, %2[%5, %5] {permutation_map = affine_map<(d0, d1) -> ()>} : vector<5x24xi8>, memref<?x?xvector<4x3xi32>>
// CHECK-NEXT: vector.transfer_write %22, %2[%5, %5] : vector<5x24xi8>, memref<?x?xvector<4x3xi32>>

vector.transfer_write %23, %3[%5, %5] {permutation_map = affine_map<(d0, d1) -> ()>} : vector<5x48xi8>, memref<?x?xvector<4x3xindex>>
// CHECK-NEXT: vector.transfer_write %23, %3[%5, %5] : vector<5x48xi8>, memref<?x?xvector<4x3xindex>>

vector.transfer_write %24, %0[%5, %5], %14 {permutation_map = affine_map<(d0, d1) -> (d1)>} : vector<5xf32>, memref<?x?xf32>
// CHECK-NEXT: vector.transfer_write %24, %0[%5, %5], %14 : vector<5xf32>, memref<?x?xf32>

// -----
// Vector transfer ops tensor
// func vector_transfer_ops_tensor in mlir/test/Dialect/Vector/ops.mlir

%0, %1, %2, %3 = "test.op"() : () -> (tensor<?x?xf32>, tensor<?x?xvector<4x3xf32>>, tensor<?x?xvector<4x3xi32>>, tensor<?x?xvector<4x3xindex>>)
// CHECK:      %0, %1, %2, %3 = "test.op"() : () -> (tensor<?x?xf32>, tensor<?x?xvector<4x3xf32>>, tensor<?x?xvector<4x3xi32>>, tensor<?x?xvector<4x3xindex>>)

%4, %5, %6, %7, %8 = "test.op"() : () -> (index, f32, f32, i32, index)
// CHECK-NEXT: %4, %5, %6, %7, %8 = "test.op"() : () -> (index, f32, f32, i32, index)

%9, %10, %11 = "test.op"() : () -> (vector<4x3xf32>, vector<4x3xi32>, vector<4x3xindex>)
// CHECK-NEXT: %9, %10, %11 = "test.op"() : () -> (vector<4x3xf32>, vector<4x3xi32>, vector<4x3xindex>)

%12 = vector.transfer_read %0[%4, %4], %6 {permutation_map = affine_map<(d0, d1) -> (d0)>, in_bounds = [false]} : tensor<?x?xf32>, vector<128xf32>
// CHECK-NEXT: %12 = vector.transfer_read %0[%4, %4], %6 {permutation_map = affine_map<(d0, d1) -> (d0)>} : tensor<?x?xf32>, vector<128xf32>

%13 = vector.transfer_read %0[%4, %4], %6 {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [false, false]} : tensor<?x?xf32>, vector<3x7xf32>
// CHECK-NEXT: %13 = vector.transfer_read %0[%4, %4], %6 {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : tensor<?x?xf32>, vector<3x7xf32>

%14 = vector.transfer_read %0[%4, %4], %5 {permutation_map = affine_map<(d0, d1) -> (d0)>, in_bounds = [false]} : tensor<?x?xf32>, vector<128xf32>
// CHECK-NEXT: %14 = vector.transfer_read %0[%4, %4], %5 {permutation_map = affine_map<(d0, d1) -> (d0)>} : tensor<?x?xf32>, vector<128xf32>

%15 = vector.transfer_read %0[%4, %4], %5 {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [false]} : tensor<?x?xf32>, vector<128xf32>
// CHECK-NEXT: %15 = vector.transfer_read %0[%4, %4], %5 : tensor<?x?xf32>, vector<128xf32>

%16 = vector.transfer_read %1[%4, %4], %9 {permutation_map = affine_map<(d0, d1) -> (d0, d1)>, in_bounds = [false, false]} : tensor<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
// CHECK-NEXT: %16 = vector.transfer_read %1[%4, %4], %9 : tensor<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>

%17 = vector.transfer_read %1[%4, %4], %9 {permutation_map = affine_map<(d0, d1) -> (d0, d1)>, in_bounds = [false, true]} : tensor<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
// CHECK-NEXT: %17 = vector.transfer_read %1[%4, %4], %9 {in_bounds = [false, true]} : tensor<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>

%18 = vector.transfer_read %2[%4, %4], %10 {permutation_map = affine_map<(d0, d1) -> ()>} : tensor<?x?xvector<4x3xi32>>, vector<5x24xi8>
// CHECK-NEXT: %18 = vector.transfer_read %2[%4, %4], %10 : tensor<?x?xvector<4x3xi32>>, vector<5x24xi8>

%19 = vector.transfer_read %3[%4, %4], %11 {permutation_map = affine_map<(d0, d1) -> ()>} : tensor<?x?xvector<4x3xindex>>, vector<5x48xi8>
// CHECK-NEXT: %19 = vector.transfer_read %3[%4, %4], %11 : tensor<?x?xvector<4x3xindex>>, vector<5x48xi8>

%20 = vector.transfer_write %12, %0[%4, %4] {permutation_map = affine_map<(d0, d1) -> (d0)>, in_bounds = [false]} : vector<128xf32>, tensor<?x?xf32>
// CHECK-NEXT: %20 = vector.transfer_write %12, %0[%4, %4] {permutation_map = affine_map<(d0, d1) -> (d0)>} : vector<128xf32>, tensor<?x?xf32>

%21 = vector.transfer_write %13, %0[%4, %4] {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [false, false]} : vector<3x7xf32>, tensor<?x?xf32>
// CHECK-NEXT: %21 = vector.transfer_write %13, %0[%4, %4] {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : vector<3x7xf32>, tensor<?x?xf32>

%22 = vector.transfer_write %16, %1[%4, %4] {permutation_map = affine_map<(d0, d1) -> (d0, d1)>, in_bounds = [false, false]} : vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>
// CHECK-NEXT: %22 = vector.transfer_write %16, %1[%4, %4] : vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>

%23 = vector.transfer_write %17, %1[%4, %4] {permutation_map = affine_map<(d0, d1) -> (d0, d1)>, in_bounds = [false, false]} : vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>
// CHECK-NEXT: %23 = vector.transfer_write %17, %1[%4, %4] : vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>

%24 = vector.transfer_write %18, %2[%4, %4] {permutation_map = affine_map<(d0, d1) -> ()>} : vector<5x24xi8>, tensor<?x?xvector<4x3xi32>>
// CHECK-NEXT: %24 = vector.transfer_write %18, %2[%4, %4] : vector<5x24xi8>, tensor<?x?xvector<4x3xi32>>

%25 = vector.transfer_write %19, %3[%4, %4] {permutation_map = affine_map<(d0, d1) -> ()>} : vector<5x48xi8>, tensor<?x?xvector<4x3xindex>>
// CHECK-NEXT: %25 = vector.transfer_write %19, %3[%4, %4] : vector<5x48xi8>, tensor<?x?xvector<4x3xindex>>
