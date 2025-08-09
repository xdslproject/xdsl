// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s
"builtin.module"() ({
  %lb = "arith.constant"() {"value" = 0 : index} : () -> index
  %ub = "arith.constant"() {"value" = 42 : index} : () -> index
  %step = "arith.constant"() {"value" = 7 : index} : () -> index
  %carried = "arith.constant"() {"value" = 255 : i8} : () -> i8
  "scf.for"(%lb, %ub, %step, %carried) ({
// CHECK: Block arg #1 expected to be i8, but got index
  ^bb0(%iv : index, %carried_arg : index):
    "scf.yield"() : () -> ()
  }) : (index, index, index, i8) -> ()
}) : () -> ()
