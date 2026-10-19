// REQUIRES: MLIR_TRANSLATE, LLVM_DIFF
// RUN: xdsl-opt -t llvm %S/../../../../backend/llvm/convert_op.mlir > %t.xdsl.ll
// RUN: $XDSL_MLIR_TRANSLATE --mlir-to-llvmir %S/../../../../backend/llvm/convert_op.mlir > %t.mlir.ll
// RUN: $XDSL_LLVM_DIFF %t.xdsl.ll %t.mlir.ll
