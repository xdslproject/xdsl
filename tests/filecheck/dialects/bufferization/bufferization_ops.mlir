// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK:       builtin.module {

// CHECK-NEXT:    %i0, %i1 = "test.op"() : () -> (index, index)
// CHECK-NEXT:    %t0 = "test.op"() : () -> tensor<10x20x30xf64>
%i0, %i1 = "test.op"() : () -> (index, index)
%t0 = "test.op"() : () -> tensor<10x20x30xf64>

// CHECK-NEXT:    %t1 = bufferization.alloc_tensor(%i0, %i1) {"hello" = "world"} : tensor<10x20x?x?xf64>
// CHECK-NEXT:    %t2 = bufferization.alloc_tensor() copy(%t0) : tensor<10x20x30xf64>
// CHECK-NEXT:    %t3 = bufferization.alloc_tensor(%i0, %i1) size_hint = %i1 : tensor<10x20x?x?xf64>
%t1 = bufferization.alloc_tensor(%i0, %i1) {"hello"="world"}: tensor<10x20x?x?xf64>
%t2 = bufferization.alloc_tensor() copy(%t0) : tensor<10x20x30xf64>
%t3 = bufferization.alloc_tensor(%i0, %i1) size_hint = %i1: tensor<10x20x?x?xf64>

// CHECK-NEXT:  }


// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    %i0, %i1 = "test.op"() : () -> (index, index)
// CHECK-GENERIC-NEXT:    %t0 = "test.op"() : () -> tensor<10x20x30xf64>
// CHECK-GENERIC-NEXT:    %t1 = "bufferization.alloc_tensor"(%i0, %i1) <{"operandSegmentSizes" = array<i32: 2, 0, 0>}> {"hello" = "world"} : (index, index) -> tensor<10x20x?x?xf64>
// CHECK-GENERIC-NEXT:    %t2 = "bufferization.alloc_tensor"(%t0) <{"operandSegmentSizes" = array<i32: 0, 1, 0>}> : (tensor<10x20x30xf64>) -> tensor<10x20x30xf64>
// CHECK-GENERIC-NEXT:    %t3 = "bufferization.alloc_tensor"(%i0, %i1, %i1) <{"operandSegmentSizes" = array<i32: 2, 0, 1>}> : (index, index, index) -> tensor<10x20x?x?xf64>
// CHECK-GENERIC-NEXT:  }) : () -> ()
