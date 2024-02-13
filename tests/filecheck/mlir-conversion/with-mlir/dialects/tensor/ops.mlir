// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

%t1 = tensor.empty() : tensor<2x3xf32>
%t2 = tensor.empty() : tensor<2xf32>
%i1 = "test.op"() : () -> (index)
%t3 = tensor.empty(%i1) : tensor<?xf32>


// CHECK:       module {
// CHECK-NEXT:  %0 = tensor.empty() :  tensor<2x3xf32>
// CHECK-NEXT:  %1 = tensor.empty() : tensor<2xf32>
// CHECK-NEXT:  %2 = "test.op"() : () -> index
// CHECK-NEXT:  %3 =  tensor.empty(%2) : tensor<?xf32>
// CHECK-NEXT: }
