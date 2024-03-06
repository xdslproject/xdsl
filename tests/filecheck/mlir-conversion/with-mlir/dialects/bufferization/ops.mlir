// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic | xdsl-opt | filecheck %s

module{
    %t = "bufferization.alloc_tensor"() <{"operandSegmentSizes" = array<i32: 0, 0, 0>}> : () -> tensor<10x20x30xf64>
    %m = "test.op"() : () -> memref<30x20x10xf32>
    %m_t = "bufferization.to_tensor"(%m) : (memref<30x20x10xf32>) -> tensor<30x20x10xf32>
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "bufferization.alloc_tensor"() <{"operandSegmentSizes" = array<i32: 0, 0, 0>}> : () -> tensor<10x20x30xf64>
// CHECK-NEXT:    %1 = "test.op"() : () -> memref<30x20x10xf32>
// CHECK-NEXT:    %2 = "bufferization.to_tensor"(%1) : (memref<30x20x10xf32>) -> tensor<30x20x10xf32>
// CHECK-NEXT:  }
