// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 0 : index} : () -> index
  %1 = "arith.constant"() {"value" = 1000 : index} : () -> index
  %2 = "arith.constant"() {"value" = 3 : index} : () -> index
  "scf.parallel"(%0, %1, %2) ({
  ^bb0(%i: index, %j: index):
    "scf.yield"() : () -> ()
  }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
}) : () -> ()

// CHECK: Expected 1 block arguments, got 2 instead
