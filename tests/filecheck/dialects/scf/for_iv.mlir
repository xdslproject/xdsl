// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s
"builtin.module"() ({
  %lb = "arith.constant"() {"value" = 0 : index} : () -> index
  %ub = "arith.constant"() {"value" = 42 : index} : () -> index
  %step = "arith.constant"() {"value" = 7 : index} : () -> index
// CHECK: Wrong number of block arguments, expected 1, got 0. The body must have the induction variable and loop-carried variables as arguments.
  "scf.for"(%lb, %ub, %step) ({
  ^0():
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
}) : () -> ()
