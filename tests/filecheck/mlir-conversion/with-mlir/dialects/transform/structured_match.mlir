// RUN: xdsl-opt %s | $XDSL_MLIR_OPT | xdsl-opt | filecheck %s

// Checks that the custom syntax xDSL prints for `transform.structured.match`
// is accepted by mlir-opt, and that the syntax mlir-opt prints is accepted by xDSL.

builtin.module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %full = transform.structured.match ops{["linalg.matmul", "linalg.generic"]} interface{TilingInterface} attributes {foo = 1 : i64} filter_result_type = f32 filter_operand_types = [f32, f32] in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK:      builtin.module attributes {transform.with_named_sequence} {
// CHECK-NEXT:   transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:     %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:     %1 = transform.structured.match ops{["linalg.matmul", "linalg.generic"]} interface{TilingInterface} attributes {foo = 1 : i64} filter_result_type = f32 filter_operand_types = [f32, f32] in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:     transform.yield
// CHECK-NEXT:   }
// CHECK-NEXT: }
