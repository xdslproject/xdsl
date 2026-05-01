// REQUIRES: llvm-diff
// RUN: sed -E \
// RUN:     -e '/^  llvm\.func @(fma_op_f32|fma_op_f64|broadcast_f32|arg_attr_types|call_intrinsic_void)/,/^  \}/d' \
// RUN:     %S/../../../../backend/llvm/convert_op.mlir > %t.filtered.mlir
// RUN: xdsl-opt -t llvm %t.filtered.mlir > %t.xdsl.ll
// RUN: mlir-translate --mlir-to-llvmir %t.filtered.mlir > %t.mlir.ll
// RUN: %llvm-diff %t.xdsl.ll %t.mlir.ll
