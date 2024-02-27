// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

%t1 = tensor.empty() : tensor<2x3xf32>
%t2 = tensor.empty() : tensor<2xf32>
%i1 = "test.op"() : () -> (index)
%t3 = tensor.empty(%i1) : tensor<?xf32>
%src = "test.op"() {"value" = dense<1.000000e-01> : tensor<4x1xf32>} : () -> tensor<4x1xf32>
%shape = "test.op"() : () -> (tensor<1xi32>)
%t4 = tensor.reshape %src(%shape) : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>


// CHECK:       module {
// CHECK-NEXT:  %0 = tensor.empty() :  tensor<2x3xf32>
// CHECK-NEXT:  %1 = tensor.empty() : tensor<2xf32>
// CHECK-NEXT:  %2 = "test.op"() : () -> index
// CHECK-NEXT:  %3 = tensor.empty(%2) : tensor<?xf32>
// CHECK-NEXT:  %4 = "test.op"() {value = dense<1.000000e-01> : tensor<4x1xf32>} : () -> tensor<4x1xf32>
// CHECK-NEXT:  %5 = "test.op"() : () -> tensor<1xi32>
// CHECK-NEXT:  %6 = tensor.reshape %4(%5) : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>
// CHECK-NEXT: }
