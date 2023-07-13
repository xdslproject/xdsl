// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s
"builtin.module"() ({
  %lb = "arith.constant"() {"value" = 0 : index} : () -> index
  %ub = "arith.constant"() {"value" = 42 : index} : () -> index
  %step = "arith.constant"() {"value" = 7 : index} : () -> index
  %carried = "arith.constant"() {"value" = 255 : i8} : () -> i8
  "scf.for"(%lb, %ub, %step, %carried) ({
// CHECK: The scf.for's body does not end with an scf.yield. A scf.for loop with loop-carried variables must yield their values at the end of its body.
  ^0(%iv : index, %carried_arg : i8):
    "test.termop"() : () -> ()
  }) : (index, index, index, i8) -> ()
}) : () -> ()
