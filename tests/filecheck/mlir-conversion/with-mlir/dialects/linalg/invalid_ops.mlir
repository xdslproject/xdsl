// RUN: xdsl-opt --verify-diagnostics --split-input-file %s | filecheck %s


builtin.module {
  %0, %1 = "test.op"() : () -> (tensor<2x3xf32>, tensor<2x3xf32>)

  // CHECK: Input type is tensor<2x3xf32> but must be an instance of AnyFloat or IntegerType.
   %res_fill = linalg.fill ins (%0: tensor<2x3xf32>) outs (%1: tensor<2x3xf32>) -> tensor<2x3xf32>

}

// -----

builtin.module {
  %0, %1 = "test.op"() : () -> (tensor<16x64xf32>, tensor<64x16x1xf32>)

  // CHECK: Operation does not verify: Input rank (2) does not match output rank (3)
	%res_transpose = linalg.transpose ins(%0 : tensor<16x64xf32>) outs(%1 : tensor<64x16x1xf32>) permutation = [1, 0]

}

// -----

builtin.module {
  %0, %1 = "test.op"() : () -> (tensor<16x64xf32>, tensor<64x16xf32>)

  // CHECK: Operation does not verify: Input rank (2) does not match size of permutation (3)
	%res_transpose = linalg.transpose ins(%0 : tensor<16x64xf32>) outs(%1 : tensor<64x16xf32>) permutation = [1, 2, 3]

}

// -----

builtin.module {
  %0, %1 = "test.op"() : () -> (tensor<16x32x64xf32>, tensor<32x64x16xf32>)

  // CHECK:  Operation does not verify: dim(result, 1) = 64 doesn't match dim(input, permutation[1]) = 32
	%res_transpose = linalg.transpose ins(%0 : tensor<16x32x64xf32>) outs(%1 : tensor<32x64x16xf32>) permutation = [1, 1, 2]

}

// -----

builtin.module {
  %0, %1 = "test.op"() : () -> (tensor<16xf32>, tensor<32x64x16xf32>)
  // CHECK:  Operation does not verify: Input rank plus added dimensions (2) does not match output rank (3)
	%res_broadcast = linalg.broadcast ins(%0 : tensor<16xf32>) outs(%1 : tensor<32x64x16xf32>) dimensions = [1]

}

// -----

builtin.module {
  %0, %1 = "test.op"() : () -> (tensor<16xf32>, tensor<16x64xf32>)
  // CHECK:  Operation does not verify: Dimension 0 is out of range.  Expected range: [0, 1], got: 9
	%res_broadcast = linalg.broadcast ins(%0 : tensor<16xf32>) outs(%1 : tensor<16x64xf32>) dimensions = [9]

}

// -----

builtin.module {
  %0, %1 = "test.op"() : () -> (tensor<3x4x5xf32>, tensor<4x5x6x2xf32>)
  // CHECK:  Operation does not verify: input dimension 0 should match output dimension 0. input: 3, output: 4
	%res_broadcast = linalg.broadcast ins(%0 : tensor<3x4x5xf32>) outs(%1 : tensor<4x5x6x2xf32>) dimensions = [1]

}
