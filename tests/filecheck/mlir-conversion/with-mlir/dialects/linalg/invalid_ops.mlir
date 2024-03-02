// RUN: xdsl-opt --verify-diagnostics --split-input-file %s | filecheck %s


builtin.module {
  %0, %1 = "test.op"() : () -> (tensor<2x3xf32>, tensor<2x3xf32>)

  // CHECK: Input type is tensor<2x3xf32> but must be an instance of AnyFloat or IntegerType.
   %res_fill = "linalg.fill"(%0, %1) <{"operandSegmentSizes" = array<i32: 1, 1>}> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>

}

// -----

builtin.module {
  %0, %1 = "test.op"() : () -> (tensor<16x64xf32>, tensor<64x16x1xf32>)

  // CHECK: Input rank (2) does not match init rank (3)
   %res_transpose = "linalg.transpose"(%0, %1) {"permutation" = array<i64: 1, 0>} : (tensor<16x64xf32>, tensor<64x16x1xf32>) -> tensor<64x16x1xf32>

}

// -----

builtin.module {
  %0, %1 = "test.op"() : () -> (tensor<16x64xf32>, tensor<64x16xf32>)

  // CHECK: Input rank (2) does not match size of permutation (3)
   %res_transpose = "linalg.transpose"(%0, %1) {"permutation" = array<i64: 1, 2, 3>} : (tensor<16x64xf32>, tensor<64x16xf32>) -> tensor<64x16xf32>

}

// -----

builtin.module {
  %0, %1 = "test.op"() : () -> (tensor<16x32x64xf32>, tensor<32x64x16xf32>)

  // CHECK: of
  %res_transpose = "linalg.transpose"(%0, %1) {"permutation" = array<i64: 1, 1, 2>} : (tensor<16x32x64xf32>, tensor<32x64x16xf32>) -> tensor<32x64x16xf32>

}






