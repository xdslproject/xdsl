// RUN: xdsl-opt %s -p lift-arith-to-linalg | filecheck %s

builtin.module {
// CHECK: builtin.module {

  func.func @a() {
    %t0, %t1, %t2, %t3 = "test.op"() : () -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
    %0 = arith.addf %t0, %t1 : tensor<8xf32>
    %1 = arith.subf %0, %t2 : tensor<8xf32>
    %2 = arith.mulf %1, %t3 : tensor<8xf32>
    func.return
  }

// CHECK-NEXT: func.func @a() {
// CHECK-NEXT:   %t0, %t1, %t2, %t3 = "test.op"() : () -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
// CHECK-NEXT:   %0 = linalg.add ins(%t0, %t1 : tensor<8xf32>, tensor<8xf32>) outs(%t0 : tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:   %1 = linalg.sub ins(%0, %t2 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:   %2 = linalg.mul ins(%1, %t3 : tensor<8xf32>, tensor<8xf32>) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:   func.return
// CHECK-NEXT: }


  func.func @choose_correct_tensor() {
    // `to_tensor`s that are not writable should not be used in `outs`
    %t0, %t1, %t2, %m0 = "test.op"() : () -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, memref<8xf32>)
    %0 = bufferization.to_tensor %m0 restrict : memref<8xf32>
    %1 = bufferization.to_tensor %m0 restrict writable : memref<8xf32>
    %2 = bufferization.to_tensor %m0 restrict : memref<8xf32>
    %3 = arith.addf %t0, %t1 : tensor<8xf32>
    %4 = arith.mulf %3, %t2 : tensor<8xf32>
    func.return
  }

// CHECK-NEXT: func.func @choose_correct_tensor() {
// CHECK-NEXT:   %t0, %t1, %t2, %m0 = "test.op"() : () -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, memref<8xf32>)
// CHECK-NEXT:   %0 = bufferization.to_tensor %m0 restrict : memref<8xf32>
// CHECK-NEXT:   %1 = bufferization.to_tensor %m0 restrict writable : memref<8xf32>
// CHECK-NEXT:   %2 = bufferization.to_tensor %m0 restrict : memref<8xf32>
// CHECK-NEXT:   %0 = linalg.add ins(%t0, %t1 : tensor<8xf32>, tensor<8xf32>) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:   %1 = linalg.mul ins(%0, %t2 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:   func.return
// CHECK-NEXT: }

  func.func @extract_slice() {
    // check that we're using the correct `extract_slice`
    %t0, %t1, %t2, %m0 = "test.op"() : () -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, memref<16xf32>)
    %tensor = bufferization.to_tensor %m0 restrict writable : memref<16xf32>
    %slice = "tensor.extract_slice"(%tensor) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 8>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<16xf32>) -> tensor<8xf32>
    %wrong_size_slice = "tensor.extract_slice"(%tensor) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 10>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<16xf32>) -> tensor<10xf32>
    %3 = arith.addf %t0, %t1 : tensor<8xf32>
    %4 = arith.mulf %3, %t2 : tensor<8xf32>
    func.return
  }

// CHECK-NEXT: func.func @extract_slice() {
// CHECK-NEXT:   %t0, %t1, %t2, %m0 = "test.op"() : () -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, memref<16xf32>)
// CHECK-NEXT:   %tensor = bufferization.to_tensor %m0 restrict writable : memref<16xf32>
// CHECK-NEXT:   %slice = "tensor.extract_slice"(%tensor) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 8>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<16xf32>) -> tensor<8xf32>
// CHECK-NEXT:   %wrong_size_slice = "tensor.extract_slice"(%tensor) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 10>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<16xf32>) -> tensor<10xf32>
// CHECK-NEXT:   %0 = linalg.add ins(%t0, %t1 : tensor<8xf32>, tensor<8xf32>) outs(%slice : tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:   %1 = linalg.mul ins(%0, %t2 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:   func.return
// CHECK-NEXT: }

}
// CHECK-NEXT: }
