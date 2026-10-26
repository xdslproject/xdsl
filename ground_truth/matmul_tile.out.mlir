module attributes {transform.with_named_sequence} {
  func.func @matmul(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c128_2 = arith.constant 128 : index
    %c128_3 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c32_4 = arith.constant 32 : index
    %c32_5 = arith.constant 32 : index
    %0 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %arg2) -> (tensor<128x128xf32>) {
      %1 = scf.for %arg5 = %c0_0 to %c128_2 step %c32_4 iter_args(%arg6 = %arg4) -> (tensor<128x128xf32>) {
        %2 = scf.for %arg7 = %c0_1 to %c128_3 step %c32_5 iter_args(%arg8 = %arg6) -> (tensor<128x128xf32>) {
          %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg7] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
          %extracted_slice_6 = tensor.extract_slice %arg1[%arg7, %arg5] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
          %extracted_slice_7 = tensor.extract_slice %arg8[%arg3, %arg5] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
          %3 = linalg.matmul ins(%extracted_slice, %extracted_slice_6 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice_7 : tensor<32x32xf32>) -> tensor<32x32xf32>
          %inserted_slice = tensor.insert_slice %3 into %arg8[%arg3, %arg5] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x128xf32>
          scf.yield %inserted_slice : tensor<128x128xf32>
        }
        scf.yield %2 : tensor<128x128xf32>
      }
      scf.yield %1 : tensor<128x128xf32>
    }
    return %0 : tensor<128x128xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [32, 32, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield 
  }
}

