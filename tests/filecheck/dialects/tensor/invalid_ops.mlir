// RUN: xdsl-opt --verify-diagnostics --split-input-file %s | filecheck %s

// Verification checks

builtin.module {
  %t0, %t1 = "test.op"() : () -> (tensor<4x1xf32>, tensor<1x2xi32>)

  // CHECK: Operation does not verify: shape tensor must have a rank one
  %res_reshape =  tensor.reshape %t0(%t1)  : (tensor<4x1xf32>, tensor<1x2xi32>) -> tensor<4xf32>
}

// -----

builtin.module {
  %t0, %t1 = "test.op"() : () -> (tensor<4x1xf32>, tensor<1xi32>)

  // CHECK: Operation does not verify: source and result tensor should have the same number of elements
  %res_reshape =  tensor.reshape %t0(%t1)  : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4x2xf32>
}

// -----

builtin.module {
  %t0, %t1 = "test.op"() : () -> (tensor<4x1xf32>, tensor<1xi32>)

  // CHECK: Operation does not verify: length of shape operand differs from the result's tensor rank
  %res_reshape =  tensor.reshape %t0(%t1)  : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<2x2xf32>
}

// -----

builtin.module {
  %t0, %t1 = "test.op"() : () -> (tensor<4x1xf64>, tensor<1xi32>)

  // CHECK: Operation does not verify: element types of source and result tensor types should be the same
  %res_reshape =  tensor.reshape %t0(%t1)  : (tensor<4x1xf64>, tensor<1xi32>) -> tensor<4xf32>
}

// -----

builtin.module {
  %t0, %t1 = "test.op"() : () -> (f32, tensor<1xi32>)

  // CHECK: operand at position 0 does not verify:
  // CHECK: f32 should be of base attribute tensor
  %res_reshape =  tensor.reshape %t0(%t1)  : (f32, tensor<1xi32>) -> tensor<4xf32>

}

// -----

builtin.module {
  %0 = "test.op"() : () -> (tensor<?x?x?xf32>)
  // CHECK:  Operation does not verify: source and destination rank should be the same
  %1 = "tensor.cast"(%0) : (tensor<?x?x?xf32>) -> tensor<?x?xf32>
}

// -----

builtin.module {
  %0 = "test.op"() : () -> (tensor<1x?xf32>)
  // CHECK:  Operation does not verify: source and destination constant dimensions should match
  %1 = "tensor.cast"(%0) : (tensor<1x?xf32>) -> tensor<2x?xf32>
}

// -----

builtin.module {
  %t, %i = "test.op"() : () -> (tensor<f32>, index)
  // CHECK:  Operation does not verify: cannot get dim of 0-rank tensor
  %1 = tensor.dim %t, %i : tensor<f32>
}
