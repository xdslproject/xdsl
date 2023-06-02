// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s
"builtin.module"() ({
  %lb = "arith.constant"() {"value" = 0 : index} : () -> index
  %ub = "arith.constant"() {"value" = 42 : index} : () -> index
  %step = "arith.constant"() {"value" = 7 : index} : () -> index
  %carried = "arith.constant"() {"value" = 255 : i8} : () -> i8
  "scf.for"(%lb, %ub, %step, %carried) ({
// CHECK: Wrong number of block arguments, expected 2, got 1. The body must have the induction variable and loop-carried variables as arguments.
  ^0(%iv : index):
    "scf.yield"() : () -> ()
  }) : (index, index, index, i8) -> ()
}) : () -> ()
