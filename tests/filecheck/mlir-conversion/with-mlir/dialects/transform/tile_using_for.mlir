// RUN: xdsl-opt %s | $XDSL_MLIR_OPT | xdsl-opt | filecheck %s

// Checks that the custom syntax xDSL prints for `transform.structured.tile_using_for`
// is accepted by mlir-opt, and that the syntax mlir-opt prints is accepted by xDSL.

builtin.module attributes {transform.with_named_sequence} {
  transform.named_sequence @tile(%matmul: !transform.any_op {transform.consumed}, %dyn: !transform.any_op {transform.readonly}) {
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul tile_sizes [32, 32, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_ij, %loops_ij:2 = transform.structured.tile_using_for %matmul tile_sizes [32, 32, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_dyn, %loops_dyn:3 = transform.structured.tile_using_for %matmul tile_sizes [%dyn, [4], 8] interchange = [1, 0, 2] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK:      builtin.module attributes {transform.with_named_sequence} {
// CHECK-NEXT:   transform.named_sequence @tile(%arg0: !transform.any_op {transform.consumed}, %arg1: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:     %tiled_linalg_op, %loops, %loops_1, %loops_2 = transform.structured.tile_using_for %arg0 tile_sizes [32, 32, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
// CHECK-NEXT:     %tiled_linalg_op_1, %loops_3, %loops_4 = transform.structured.tile_using_for %arg0 tile_sizes [32, 32, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
// CHECK-NEXT:     %tiled_linalg_op_2, %loops_5, %loops_6, %loops_7 = transform.structured.tile_using_for %arg0 tile_sizes [%arg1, [4], 8] interchange = [1, 0, 2] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
// CHECK-NEXT:     transform.yield
// CHECK-NEXT:   }
// CHECK-NEXT: }
