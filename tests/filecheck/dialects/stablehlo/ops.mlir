// RUN: XDSL_ROUNDTRIP

builtin.module {
  %t0 = "test.op"() : () -> tensor<i32>

  // CHECK: %abs = "stablehlo.abs"(%t0) : (tensor<i32>) -> tensor<i32>
  %abs = "stablehlo.abs"(%t0) : (tensor<i32>) -> tensor<i32>

  // CHECK: %add = "stablehlo.add"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %add = "stablehlo.add"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  // CHECK: %multiply = "stablehlo.multiply"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %multiply = "stablehlo.multiply"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  // CHECK: %subtract = "stablehlo.subtract"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %subtract = "stablehlo.subtract"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
}
