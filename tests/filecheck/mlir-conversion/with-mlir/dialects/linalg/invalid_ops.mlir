// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s


builtin.module {
  %0, %1 = "test.op"() : () -> (tensor<2x3xf32>, tensor<2x3xf32>)
  // CHECK: Input type is tensor<2x3xf32> but must be an instance of AnyFloat or IntegerType.
  %fill = "linalg.fill"(%0, %1) <{"operandSegmentSizes" = array<i32: 1, 1>}> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
}
