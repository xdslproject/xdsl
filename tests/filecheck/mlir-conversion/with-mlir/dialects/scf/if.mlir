// RUN: xdsl-opt %s -t mlir | mlir-opt --mlir-print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = false} : () -> i1
  "scf.if"(%0) ({}, {}) : (i1) -> ()
}) : () -> ()

// CHECK: "scf.if"({{%\d+}}) ({}, {}) : (i1) -> ()


