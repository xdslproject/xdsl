// RUN: xdsl-opt %S/../../../dialects/polynomial/ops.mlir | heir-opt --allow-unregistered-dialect | xdsl-opt | filecheck %S/../../../dialects/polynomial/ops.mlir
// RUN: xdsl-opt %S/../../../dialects/polynomial/ops.mlir --print-op-generic | heir-opt --allow-unregistered-dialect | xdsl-opt | filecheck %S/../../../dialects/polynomial/ops.mlir

// RUN: heir-opt --allow-unregistered-dialect %S/../../../dialects/polynomial/ops.mlir | xdsl-opt | filecheck %S/../../../dialects/polynomial/ops.mlir
// RUN: heir-opt --allow-unregistered-dialect --mlir-print-op-generic %S/../../../dialects/polynomial/ops.mlir | xdsl-opt | filecheck %S/../../../dialects/polynomial/ops.mlir
