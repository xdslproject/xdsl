// RUN: python -m toy %s --emit=shape-inference --ir | filecheck %s

"builtin.module"() ({
  "toy.func"() ({
    %0 = "toy.constant"() {"value" = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>

    %1 = "toy.cast"(%0) : (tensor<2x3xf64>) -> tensor<*xf64>
    %2 = "toy.transpose"(%1) : (tensor<*xf64>) -> tensor<*xf64>
    %3 = "toy.mul"(%2, %2) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    "toy.print"(%3) : (tensor<*xf64>) -> ()

// CHECK:       %1 = "toy.transpose"(%0) : (tensor<2x3xf64>) -> tensor<3x2xf64>
// CHECK-NEXT:  %2 = "toy.mul"(%1, %1) : (tensor<3x2xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
// CHECK-NEXT:  "toy.print"(%2) : (tensor<3x2xf64>) -> ()

    "toy.return"() : () -> ()
  }) {"sym_name" = "main", "function_type" = () -> ()} : () -> ()
}) : () -> ()
