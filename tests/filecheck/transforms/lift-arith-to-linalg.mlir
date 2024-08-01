// RUN: xdsl-opt %s -p lift-arith-to-linalg | filecheck %s

builtin.module {
    %t0, %t1, %t2, %t3 = "test.op"() : () -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
    %0 = arith.addf %t0, %t1 : tensor<8xf32>
    %1 = arith.subf %0, %t2 : tensor<8xf32>
    %2 = arith.mulf %1, %t3 : tensor<8xf32>
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %t0, %t1, %t2, %t3 = "test.op"() : () -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
// CHECK-NEXT:   %0 = linalg.add ins(%t0, %t1 : tensor<8xf32>, tensor<8xf32>) outs(%t0 : tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:   %1 = linalg.sub ins(%0, %t2 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:   %2 = linalg.mul ins(%1, %t3 : tensor<8xf32>, tensor<8xf32>) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT: }
