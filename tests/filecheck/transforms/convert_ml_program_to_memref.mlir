// RUN: xdsl-opt %s -p convert-ml-program-to-memref | filecheck %s

ml_program.global private @global_same_type(dense<4> : tensor<4xi32>) : tensor<4xi32>

%0 = ml_program.global_load_const @global_same_type : tensor<4xi32>

// CHECK:       builtin.module {
// CHECK-NEXT:    memref.global constant "private" @global_same_type : memref<4xi32> = dense<4>
// CHECK-NEXT:    %0 = memref.get_global @global_same_type : memref<4xi32>
// CHECK-NEXT:    %1 = bufferization.to_tensor %0 : memref<4xi32>
// CHECK-NEXT:  }
