// RUN: xdsl-opt %s | xdsl-opt | filecheck %s

%0, %1 = "test.op"() : () -> (memref<64x9216xf32>, memref<9216x4096xf32>)
%2 = "test.op"() : () -> (memref<64x4096xf32>)
linalg.matmul {id} ins(%0, %1 : memref<64x9216xf32>, memref<9216x4096xf32>) outs(%2 : memref<64x4096xf32>)

// CHECK-NEXT:  module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (memref<64x9216xf32>, memref<9216x4096xf32>)
// CHECK-NEXT:    %2 = "test.op"() : () -> memref<64x4096xf32>
// CHECK-NEXT:    linalg.matmul {"id"} ins(%0, %1 : memref<64x9216xf32>, memref<9216x4096xf32>) outs(%2 : memref<64x4096xf32>)
// CHECK-NEXT:  }
