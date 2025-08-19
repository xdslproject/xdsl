// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  %lb = "arith.constant"() {"value" = 0 : index} : () -> index
  %ub = "arith.constant"() {"value" = 42 : index} : () -> index
  %step = "arith.constant"() {"value" = 7 : index} : () -> index
// CHECK: Expected at least 3 operands, but got 2
  "scf.for"(%ub, %step) ({
  ^bb0(%iv : index):
    "scf.yield"() : () -> ()
  }) : (index, index) -> ()
}) : () -> ()
