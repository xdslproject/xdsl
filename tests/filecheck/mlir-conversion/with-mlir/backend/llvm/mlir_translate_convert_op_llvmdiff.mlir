// REQUIRES: LLVM_DIFF
// RUN: xdsl-opt -t llvm %S/../../../../backend/llvm/convert_op.mlir > %t.xdsl.ll
// RUN: mlir-translate --mlir-to-llvmir %S/../../../../backend/llvm/convert_op.mlir > %t.mlir.ll
// RUN: llvm-diff %t.xdsl.ll %t.mlir.ll
