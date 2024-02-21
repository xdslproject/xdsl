// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s


ml_program.global private @global_same_type(dense<4> : tensor<4xi32>) : tensor<4xi32>

// CHECK:       module {
// CHECK-NEXT: ml_program.global private @global_same_type(dense<4> : tensor<4xi32>) : tensor<4xi32>
// CHECK-NEXT: }


