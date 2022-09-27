"builtin.module"() ({
  "toy.func"() ({
  ^bb0(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>):
    %0 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64>
    %1 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64>
    %2 = "toy.mul"(%0, %1) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    "toy.return"(%2) : (tensor<*xf64>) -> ()
  }) {function_type = (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>, sym_name = "multiply_transpose", sym_visibility = "private"} : () -> ()
  "toy.func"() ({
    %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %1 = "toy.reshape"(%0) : (tensor<2x3xf64>) -> tensor<2x3xf64>
    %2 = "toy.constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : () -> tensor<6xf64>
    %3 = "toy.reshape"(%2) : (tensor<6xf64>) -> tensor<2x3xf64>
    %4 = "toy.generic_call"(%1, %3) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    %5 = "toy.generic_call"(%3, %1) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    "toy.print"(%5) : (tensor<*xf64>) -> ()
    "toy.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
}) : () -> ()