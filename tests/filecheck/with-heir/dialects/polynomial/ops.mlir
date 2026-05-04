// RUN: xdsl-opt %S/../../../dialects/polynomial/ops.mlir | heir-opt --allow-unregistered-dialect | xdsl-opt | filecheck %S/../../../dialects/polynomial/ops.mlir
