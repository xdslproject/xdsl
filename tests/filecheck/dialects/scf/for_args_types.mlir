// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s
"builtin.module"() ({
  %lb = "arith.constant"() {"value" = 0 : i32} : () -> index
  %ub = "arith.constant"() {"value" = 42 : index} : () -> index
  %step = "arith.constant"() {"value" = 7 : index} : () -> index
  %lbi = "arith.constant"() {"value" = 0 : i32} : () -> i32
  %ubi = "arith.constant"() {"value" = 42 : index} : () -> i32
  %stepi = "arith.constant"() {"value" = 7 : index} : () -> i32
// CHECK: i32 should be of base attribute index
  "scf.for"(%lbi, %ub, %step) ({
  ^0(%iv : index):
    "scf.yield"() : () -> ()
  }) : (i32, index, index) -> ()
}) : () -> ()
