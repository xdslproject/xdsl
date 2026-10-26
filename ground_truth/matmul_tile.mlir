// Ground truth for xdslproject/xdsl#6450, step 1.
//
// Run with LLVM 22.1.2 (the version CI pins, ghcr.io/xdslproject/llvm:22.1.2):
//
//   mlir-opt matmul_tile.mlir -transform-interpreter
//
// Expected: the linalg.matmul is replaced by three nested scf.for loops (i, j, k),
// each stepping by 32, wrapping a 32x32x32 linalg.matmul on tensor.extract_slice'd
// operands, with the result threaded through iter_args / tensor.insert_slice.
// The named sequence stays in the output; ignore it.

module attributes {transform.with_named_sequence} {
  func.func @matmul(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    // One result for the tiled op, plus one loop per non-zero tile size: 1 + 3 = 4.
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul tile_sizes [32, 32, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
