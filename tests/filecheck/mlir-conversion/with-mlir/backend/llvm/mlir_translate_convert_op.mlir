// RUN: sed -E \
// RUN:     -e '/^  llvm\.func @(fma_op_f32|fma_op_f64|broadcast_f32|arg_attr_types|call_intrinsic_void)/,/^  \}/d' \
// RUN:     -e '/CHECK.*(fma_op_f32|fma_op_f64|broadcast_f32|arg_attr_types|call_intrinsic_void)/,/CHECK-NEXT: \}/d' \
// RUN:     %S/../../../../backend/llvm/convert_op.mlir > %t.filtered.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.filtered.mlir | filecheck %t.filtered.mlir
