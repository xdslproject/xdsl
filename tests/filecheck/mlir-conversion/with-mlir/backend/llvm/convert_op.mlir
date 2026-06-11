// REQUIRES: MLIR_TRANSLATE
// RUN: xdsl-opt %S/../../../../backend/llvm/convert_op.mlir | %MLIR_TRANSLATE --mlir-to-llvmir > /dev/null
