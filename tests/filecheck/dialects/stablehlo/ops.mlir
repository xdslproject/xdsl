// RUN: XDSL_ROUNDTRIP

%t0 = "test.op"() : () -> tensor<i32>

// CHECK: %abs = "stablehlo.abs"(%t0) : (tensor<i32>) -> tensor<i32>
%abs = "stablehlo.abs"(%t0) : (tensor<i32>) -> tensor<i32>

// CHECK: %add = "stablehlo.add"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%add = "stablehlo.add"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

%token0 = "test.op"() : () -> !stablehlo.token
%token1 = "test.op"() : () -> !stablehlo.token
// CHECK: %after_all = "stablehlo.after_all"(%token0, %token1) : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token
%after_all = "stablehlo.after_all"(%token0, %token1) : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token

// CHECK: %multiply = "stablehlo.multiply"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%multiply = "stablehlo.multiply"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

// CHECK: %subtract = "stablehlo.subtract"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%subtract = "stablehlo.subtract"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

%transpose_operand = "test.op"() : () -> tensor<2x3x2xi32>
// %operand: [
//            [[1,2], [3,4], [5,6]],
//            [[7,8], [9,10], [11,12]]
//           ]
// CHECK:  %transpose_result = "stablehlo.transpose"(%transpose_operand) {permutation = array<i64: 2, 1, 0>} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
%transpose_result = "stablehlo.transpose"(%transpose_operand) {
  permutation = array<i64: 2, 1, 0>
} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
// %result: [
//           [[1,7], [3,9], [5,11]],
//           [[2,8], [4,10], [6,12]]
//          ]

// CHECK: %and = "stablehlo.and"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%and = "stablehlo.and"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

// CHECK: %or = "stablehlo.or"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%or = "stablehlo.or"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

// CHECK: %xor = "stablehlo.xor"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%xor = "stablehlo.xor"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

// %bitcast = "stablehlo.bitcast_convert"(%t0) : (tensor<i32>) -> tensor<2xi16>
%bitcast = "stablehlo.bitcast_convert"(%t0) : (tensor<i32>) -> tensor<2xi16>

%index = "test.op"() : () -> tensor<i32>
%result_branch0 = "test.op"() : () -> tensor<2xi64>
%result_branch1 = "test.op"() : () -> tensor<2xi64>

// CHECK: %0, %1 = "stablehlo.case"(%index) ({
%0:2 = "stablehlo.case"(%index) ({
  // CHECK: "stablehlo.return"(%result_branch0, %result_branch0) : (tensor<2xi64>, tensor<2xi64>) -> ()
  "stablehlo.return"(%result_branch0, %result_branch0) : (tensor<2xi64>, tensor<2xi64>) -> ()
}, {
  // CHECK: "stablehlo.return"(%result_branch1, %result_branch1) : (tensor<2xi64>, tensor<2xi64>) -> ()
  "stablehlo.return"(%result_branch1, %result_branch1) : (tensor<2xi64>, tensor<2xi64>) -> ()
// CHECK: }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
}) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
