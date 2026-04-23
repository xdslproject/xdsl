// RUN: xdsl-opt %S/../../../dialects/polynomial/attrs.mlir | heir-opt --allow-unregistered-dialect | xdsl-opt | filecheck %S/../../../dialects/polynomial/attrs.mlir
