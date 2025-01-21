"builtin.module"() ({
  %0 = "test.op"() : () -> tensor<2x3x2xi32>
  %1 = "stablehlo.transpose"(%0) {permutation = array<i64: 2, 1, 0>} : (tensor<2x3x2xi32>) -> tensor<4x3x2xi32>
}) : () -> ()
