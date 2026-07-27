// RUN: xdsl-opt %S/../../../dialects/polynomial/types.mlir | heir-opt --allow-unregistered-dialect | xdsl-opt | filecheck %S/../../../dialects/polynomial/types.mlir
