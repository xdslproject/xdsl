// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | filecheck %s


%t = bufferization.alloc_tensor() : tensor<10x20x30xf64>
%m = "test.op"() : () -> memref<30x20x10xf32>
%m_t = bufferization.to_tensor %m restrict writable : memref<30x20x10xf32> to tensor<30x20x10xf32>
%t_m = bufferization.to_buffer %m_t read_only : tensor<30x20x10xf32> to memref<30x20x10xf32>
%c = bufferization.clone %m : memref<30x20x10xf32> to memref<30x20x10xf32>

%tensor1 = "test.op"() : () -> tensor<2x2xf64>
%tensor2 = "test.op"() : () -> tensor<2x2xf64>
%memref = "test.op"() : () -> memref<2x2xf64>
bufferization.materialize_in_destination %tensor1 in writable %memref : (tensor<2x2xf64>, memref<2x2xf64>) -> ()
%b = bufferization.materialize_in_destination %tensor1 in %tensor2 : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>


// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = bufferization.alloc_tensor() : tensor<10x20x30xf64>
// CHECK-NEXT:    %1 = "test.op"() : () -> memref<30x20x10xf32>
// CHECK-NEXT:    %2 = bufferization.to_tensor %1 restrict writable : memref<30x20x10xf32> to tensor<30x20x10xf32>
// CHECK-NEXT:    %3 = bufferization.to_buffer %2 read_only : tensor<30x20x10xf32> to memref<30x20x10xf32>
// CHECK-NEXT:    %4 = bufferization.clone %1 : memref<30x20x10xf32> to memref<30x20x10xf32>
// CHECK-NEXT:    %5 = "test.op"() : () -> tensor<2x2xf64>
// CHECK-NEXT:    %6 = "test.op"() : () -> tensor<2x2xf64>
// CHECK-NEXT:    %7 = "test.op"() : () -> memref<2x2xf64>
// CHECK-NEXT:    bufferization.materialize_in_destination %5 in writable %7 : (tensor<2x2xf64>, memref<2x2xf64>) -> ()
// CHECK-NEXT:    %8 = bufferization.materialize_in_destination %5 in %6 : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:  }
