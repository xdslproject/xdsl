// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s


%t1 = tensor.empty() : tensor<2x3xf32>

// CHECK:       module {
// CHECK-NEXT:   %1 = tensor.empty() : tensor<2x3xf32>
// CHECK-NEXT: }
