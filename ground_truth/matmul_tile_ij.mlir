// Variant of matmul_tile.mlir that only tiles the parallel (i, j) dimensions
// and leaves the reduction dimension k alone. Same command:
//
//   mlir-opt matmul_tile_ij.mlir -transform-interpreter
//
// Expected: two nested scf.for loops wrapping a 32x32x128 linalg.matmul.

module attributes {transform.with_named_sequence} {
  func.func @matmul(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    // A zero tile size produces no loop, so this is 1 + 2 = 3 results.
    %tiled, %loops:2 = transform.structured.tile_using_for %matmul tile_sizes [32, 32, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
