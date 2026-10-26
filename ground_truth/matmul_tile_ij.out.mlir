module attributes {transform.with_named_sequence} {
  func.func @matmul(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c128_1 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c32_2 = arith.constant 32 : index
    %0 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %arg2) -> (tensor<128x128xf32>) {
      %1 = scf.for %arg5 = %c0_0 to %c128_1 step %c32_2 iter_args(%arg6 = %arg4) -> (tensor<128x128xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [32, 128] [1, 1] : tensor<128x128xf32> to tensor<32x128xf32>
        %extracted_slice_3 = tensor.extract_slice %arg1[0, %arg5] [128, 32] [1, 1] : tensor<128x128xf32> to tensor<128x32xf32>
        %extracted_slice_4 = tensor.extract_slice %arg6[%arg3, %arg5] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
        %2 = linalg.matmul ins(%extracted_slice, %extracted_slice_3 : tensor<32x128xf32>, tensor<128x32xf32>) outs(%extracted_slice_4 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %inserted_slice = tensor.insert_slice %2 into %arg6[%arg3, %arg5] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x128xf32>
        scf.yield %inserted_slice : tensor<128x128xf32>
      }
      scf.yield %1 : tensor<128x128xf32>
    }
    return %0 : tensor<128x128xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [32, 32, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield 
  }
}

