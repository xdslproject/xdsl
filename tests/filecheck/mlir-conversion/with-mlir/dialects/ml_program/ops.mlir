// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

%res_const = ml_program.global_load_const @global_same_type : tensor<4xi32>

// CHECK:       module {
// CHECK-NEXT: %0 = ml_program.global_load_const @global_same_type : tensor<4xi32>
// CHECK-NEXT: }


