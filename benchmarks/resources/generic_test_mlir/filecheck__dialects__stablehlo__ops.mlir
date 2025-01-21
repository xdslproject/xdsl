"builtin.module"() ({
  %0 = "test.op"() : () -> tensor<i32>
  %1 = "stablehlo.abs"(%0) : (tensor<i32>) -> tensor<i32>
  %2 = "stablehlo.add"(%0, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "test.op"() : () -> !stablehlo.token
  %4 = "test.op"() : () -> !stablehlo.token
  %5 = "stablehlo.after_all"(%3, %4) : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token
  %6 = "stablehlo.multiply"(%0, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %7 = "stablehlo.subtract"(%0, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %8 = "test.op"() : () -> tensor<2x3x2xi32>
  %9 = "stablehlo.transpose"(%8) {permutation = array<i64: 2, 1, 0>} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
  %10 = "stablehlo.and"(%0, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %11 = "stablehlo.bitcast_convert"(%0) : (tensor<i32>) -> tensor<2xi16>
  %12 = "test.op"() : () -> tensor<i32>
  %13 = "test.op"() : () -> tensor<2xi64>
  %14 = "test.op"() : () -> tensor<2xi64>
  %15:2 = "stablehlo.case"(%12) ({
    "stablehlo.return"(%13, %13) : (tensor<2xi64>, tensor<2xi64>) -> ()
  }, {
    "stablehlo.return"(%14, %14) : (tensor<2xi64>, tensor<2xi64>) -> ()
  }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
}) : () -> ()
