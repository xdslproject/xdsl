// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s
"builtin.module"() ({
  %lb = "arith.constant"() {"value" = 0 : index} : () -> index
  %ub = "arith.constant"() {"value" = 42 : index} : () -> index
  %step = "arith.constant"() {"value" = 7 : index} : () -> index
  %carried = "arith.constant"() {"value" = 36000 : i8} : () -> i8
  "scf.for"(%lb, %ub, %step, %carried) ({
// CHECK: Expected i8, got index. The scf.for's scf.yield must match carried variables types.
  ^0(%iv : index, %carried : i8):
    "scf.yield"(%iv) : () -> ()
  }) : (index, index, index) -> ()
}) : () -> ()
