// RUN: xdsl-opt %s -p convert-ml-program-to-memref | filecheck %s

ml_program.global private @global_same_type(dense<4> : tensor<4xi32>) : tensor<4xi32>

%0 = ml_program.global_load_const @global_same_type : tensor<4xi32>

// CHECK:       module {
// CHECK-NEXT:    ml_program.global private @global_same_type(dense<4> : tensor<4xi32>) : tensor<4xi32>
// CHECK-NEXT:    %global_same_type = ml_program.global_load_const @global_same_type : tensor<4xi32>
// CHECK-NEXT:  }
