// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s
"builtin.module"() ({
  %lb = "arith.constant"() {"value" = 0 : index} : () -> index
  %ub = "arith.constant"() {"value" = 42 : index} : () -> index
  %step = "arith.constant"() {"value" = 7 : index} : () -> index
  %carried = "arith.constant"() {"value" = 255 : i8} : () -> i8
  "scf.for"(%lb, %ub, %step, %carried) ({
// CHECK: 'scf.for': along control flow edge from Operation 'scf.yield' to Region #0: successor operand type #0 index should match successor input type #0 i8
  ^bb0(%iv: index, %carried_arg: i8):
    "scf.yield"(%iv) : (index) -> ()
  }) : (index, index, index, i8) -> ()
}) : () -> ()
