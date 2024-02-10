// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s


%t1 = tensor.empty() : tensor<2x3xf32>
%t2 = tensor.empty() : tensor<2xf32>
%i1 = "test.op"() : () -> index
%t3 = tensor.empty(%i1 : index) : tensor<3xf32>



// CHECK:       module {
// CHECK-NEXT:  %t1 = tensor.empty() :  tensor<2x3xf32>
// CHECK-NEXT:  %t2 = tensor.empty() : tensor<2xf32>
// CHECK-NEXT:  %i1 = "test.op"() : () -> index
// CHECK-NEXT:  %t3 = tensor.empty(%i1 : index) : tensor<3xf32>
// CHECK-NEXT: }
