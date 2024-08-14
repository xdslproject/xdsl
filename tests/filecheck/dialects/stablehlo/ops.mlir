// RUN: XDSL_ROUNDTRIP

%t0 = "test.op"() : () -> tensor<i32>

// CHECK: %abs = "stablehlo.abs"(%t0) : (tensor<i32>) -> tensor<i32>
%abs = "stablehlo.abs"(%t0) : (tensor<i32>) -> tensor<i32>

// CHECK: %add = "stablehlo.add"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%add = "stablehlo.add"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

// CHECK: %multiply = "stablehlo.multiply"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%multiply = "stablehlo.multiply"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

// CHECK: %subtract = "stablehlo.subtract"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%subtract = "stablehlo.subtract"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

%transpose_operand = "test.op"() : () -> tensor<2x3x2xi32>
// %operand: [
//            [[1,2], [3,4], [5,6]],
//            [[7,8], [9,10], [11,12]]
//           ]
%transpose_result = "stablehlo.transpose"(%transpose_operand) {
  permutation = array<i64: 2, 1, 0>
} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
// %result: [
//           [[1,7], [3,9], [5,11]],
//           [[2,8], [4,10], [6,12]]
//          ]
