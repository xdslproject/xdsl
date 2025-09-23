// RUN: python -m toy %s --emit=toy-inline --ir | filecheck %s

builtin.module {
// CHECK: builtin.module {

  "toy.func"() ({
  ^bb0(%0 : tensor<*xf64>, %1 : tensor<*xf64>):
    %2 = "toy.transpose"(%0) : (tensor<*xf64>) -> tensor<*xf64>
    %3 = "toy.transpose"(%1) : (tensor<*xf64>) -> tensor<*xf64>
    %4 = "toy.mul"(%2, %3) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    "toy.return"(%4) : (tensor<*xf64>) -> ()
  }) {sym_name = "multiply_transpose", function_type = (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>, sym_visibility = "private"} : () -> ()
  "toy.func"() ({
    %5 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %6 = "toy.reshape"(%5) : (tensor<2x3xf64>) -> tensor<2x3xf64>
    %7 = "toy.constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : () -> tensor<6xf64>
    %8 = "toy.reshape"(%7) : (tensor<6xf64>) -> tensor<2x3xf64>
    %9 = "toy.generic_call"(%6, %8) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    %10 = "toy.generic_call"(%8, %6) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    "toy.print"(%10) : (tensor<*xf64>) -> ()
    "toy.return"() : () -> ()
  }) {sym_name = "main", function_type = () -> ()} : () -> ()

// CHECK-NEXT:  "toy.func"() ({
// CHECK-NEXT{LITERAL}:    %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
// CHECK-NEXT{LITERAL}:    %1 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
// CHECK-NEXT:    %2 = "toy.cast"(%1) : (tensor<2x3xf64>) -> tensor<*xf64>
// CHECK-NEXT:    %3 = "toy.cast"(%0) : (tensor<2x3xf64>) -> tensor<*xf64>
// CHECK-NEXT:    %4 = "toy.transpose"(%2) : (tensor<*xf64>) -> tensor<*xf64>
// CHECK-NEXT:    %5 = "toy.transpose"(%3) : (tensor<*xf64>) -> tensor<*xf64>
// CHECK-NEXT:    %6 = "toy.mul"(%4, %5) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
// CHECK-NEXT:    "toy.print"(%6) : (tensor<*xf64>) -> ()
// CHECK-NEXT:    "toy.return"() : () -> ()
// CHECK-NEXT:  }) {sym_name = "main", function_type = () -> ()} : () -> ()

}
// CHECK-NEXT: }
