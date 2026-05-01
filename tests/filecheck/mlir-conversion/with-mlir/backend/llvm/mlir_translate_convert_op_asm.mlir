// REQUIRES: llc
// XFAIL: *
// xdsl backend does not emit inteldialect flag for asm_dialect = intel
// RUN: sed -E \
// RUN:     -e '/^  llvm\.func @(fma_op_f32|fma_op_f64|broadcast_f32|arg_attr_types|call_intrinsic_void)/,/^  \}/d' \
// RUN:     %S/../../../../backend/llvm/convert_op.mlir > %t.filtered.mlir
// RUN: xdsl-opt -t llvm %t.filtered.mlir | %llc -O0 -mtriple=x86_64-unknown-linux-gnu | sed -E \
// RUN:     -e 's/[[:space:]]*#.*$//' \
// RUN:     -e 's/\.file.*$/.file/' \
// RUN:     -e 's/[[:space:]]+$//' \
// RUN:     > %t.xdsl.s
// RUN: mlir-translate --mlir-to-llvmir %t.filtered.mlir | %llc -O0 -mtriple=x86_64-unknown-linux-gnu | sed -E \
// RUN:     -e 's/[[:space:]]*#.*$//' \
// RUN:     -e 's/\.file.*$/.file/' \
// RUN:     -e 's/[[:space:]]+$//' \
// RUN:     > %t.mlir.s
// RUN: diff %t.xdsl.s %t.mlir.s
