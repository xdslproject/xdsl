// RUN: xdsl-opt %S/../../../dialects/polynomial/attrs.mlir | heir-opt --allow-unregistered-dialect | xdsl-opt | filecheck %S/../../../dialects/polynomial/attrs.mlir
// RUN: heir-opt --allow-unregistered-dialect %S/../../../dialects/polynomial/attrs.mlir | xdsl-opt | filecheck %S/../../../dialects/polynomial/attrs.mlir
