// REQUIRES: MLIR_TRANSLATE
// RUN: xdsl-opt %S/../../../../backend/llvm/module.mlir | %MLIR_TRANSLATE --mlir-to-llvmir > /dev/null
