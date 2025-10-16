// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s
"builtin.module"() ({
  %lb = "arith.constant"() {"value" = 0 : index} : () -> index
  %ub = "arith.constant"() {"value" = 42 : index} : () -> index
  %step = "arith.constant"() {"value" = 7 : index} : () -> index
// CHECK: Body block must have induction var as first block arg
  "scf.for"(%lb, %ub, %step) ({
  ^bb0():
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
}) : () -> ()
