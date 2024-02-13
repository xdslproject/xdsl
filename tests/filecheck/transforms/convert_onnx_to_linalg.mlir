// RUN: xdsl-opt -p convert-onnx-to-linalg %s | filecheck %s


func.func @main_graph(%0 : tensor<3x2xf32>, %1 : tensor<3x2xf32>) -> tensor<3x2xf32> {
     %2 = tensor.empty() : tensor<3x2xf32>
     %3 = linalg.add ins(%0, %1 : tensor<3x2xf32>, tensor<3x2xf32>) outs(%2 : tensor<3x2xf32>) -> tensor<3x2xf32>
     func.return %3 : tensor<3x2xf32>
   }


// CHECK: builtin.module {
// CHECK-NEXT:   func.func @main_graph(%0 : tensor<3x2xf32>, %1 : tensor<3x2xf32>) -> tensor<3x2xf32> {
// CHECK-NEXT:     %2 = tensor.empty() : tensor<3x2xf32>
// CHECK-NEXT:     %3 = linalg.add ins(%0, %1 : tensor<3x2xf32>, tensor<3x2xf32>) outs(%2 : tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:     func.return %3 : tensor<3x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
