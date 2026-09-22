// RUN: xdsl-opt -p transform-interpreter %s | filecheck %s

// Tile a tensor matmul through the transform interpreter, using MLIR-compatible
// schedule syntax. Check all three loops and the slices and yields that carry
// the accumulator through the reduction. The schedule remains in the output.

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

// CHECK-LABEL: func.func @matmul(
// CHECK-SAME: %[[A:[a-zA-Z0-9_]+]]: tensor<128x128xf32>, %[[B:[a-zA-Z0-9_]+]]: tensor<128x128xf32>, %[[C:[a-zA-Z0-9_]+]]: tensor<128x128xf32>) -> tensor<128x128xf32> {
// CHECK-NEXT: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : index
// CHECK-NEXT: %[[MI:[a-zA-Z0-9_]+]] = arith.constant 128 : index
// CHECK-NEXT: %[[NJ:[a-zA-Z0-9_]+]] = arith.constant 128 : index
// CHECK-NEXT: %[[KK:[a-zA-Z0-9_]+]] = arith.constant 128 : index
// CHECK-NEXT: %[[TI:[a-zA-Z0-9_]+]] = arith.constant 32 : index
// CHECK-NEXT: %[[TJ:[a-zA-Z0-9_]+]] = arith.constant 32 : index
// CHECK-NEXT: %[[TK:[a-zA-Z0-9_]+]] = arith.constant 32 : index
// CHECK-NEXT: %[[OUTER:[a-zA-Z0-9_]+]] = scf.for %[[I:[a-zA-Z0-9_]+]] = %[[ZERO]] to %[[MI]] step %[[TI]] iter_args(%[[CI:[a-zA-Z0-9_]+]] = %[[C]]) -> (tensor<128x128xf32>) {
// CHECK-NEXT: %[[MIDDLE:[a-zA-Z0-9_]+]] = scf.for %[[J:[a-zA-Z0-9_]+]] = %[[ZERO]] to %[[NJ]] step %[[TJ]] iter_args(%[[CJ:[a-zA-Z0-9_]+]] = %[[CI]]) -> (tensor<128x128xf32>) {
// CHECK-NEXT: %[[INNER:[a-zA-Z0-9_]+]] = scf.for %[[K:[a-zA-Z0-9_]+]] = %[[ZERO]] to %[[KK]] step %[[TK]] iter_args(%[[CK:[a-zA-Z0-9_]+]] = %[[CJ]]) -> (tensor<128x128xf32>) {
// CHECK-NEXT: %[[AS:[a-zA-Z0-9_]+]] = tensor.extract_slice %[[A]][%[[I]], %[[K]]] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
// CHECK-NEXT: %[[BS:[a-zA-Z0-9_]+]] = tensor.extract_slice %[[B]][%[[K]], %[[J]]] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
// CHECK-NEXT: %[[CS:[a-zA-Z0-9_]+]] = tensor.extract_slice %[[CK]][%[[I]], %[[J]]] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
// CHECK-NEXT: %[[TILE:[a-zA-Z0-9_]+]] = linalg.matmul ins(%[[AS]], %[[BS]] : tensor<32x32xf32>, tensor<32x32xf32>) outs(%[[CS]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK-NEXT: %[[UPDATED:[a-zA-Z0-9_]+]] = tensor.insert_slice %[[TILE]] into %[[CK]][%[[I]], %[[J]]] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x128xf32>
// CHECK-NEXT: scf.yield %[[UPDATED]] : tensor<128x128xf32>
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield %[[INNER]] : tensor<128x128xf32>
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield %[[MIDDLE]] : tensor<128x128xf32>
// CHECK-NEXT: }
// CHECK-NEXT: func.return %[[OUTER]] : tensor<128x128xf32>
// CHECK-NEXT: }
// CHECK: transform.named_sequence @__transform_main
