// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s
"builtin.module"() ({
  %lb = "arith.constant"() {"value" = 0 : index} : () -> index
  %ub = "arith.constant"() {"value" = 42 : index} : () -> index
  %step = "arith.constant"() {"value" = 7 : index} : () -> index
  %carried = "arith.constant"() {"value" = 255 : i8} : () -> i8
  "scf.for"(%lb, %ub, %step) ({
// CHECK: Expected 1 args, but got 2. Body block must have induction and loop-carried variables as args
  ^0(%iv : index, %carried_arg : i8):
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
}) : () -> ()
