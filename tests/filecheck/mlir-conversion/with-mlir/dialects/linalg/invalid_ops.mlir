// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s


builtin.module {
  %2, %3 = "test.op"() : () -> (tensor<2x3xf32>, tensor<2x3xf32>)

  // CHECK: Input type is tensor<2x3xf32> but must be an instance of AnyFloat or IntegerType.
  %fill = linalg.fill ins(%2 : tensor<2x3xf32>) outs(%3 : tensor<2x3xf32>) -> tensor<2x3xf32>
}
