// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 0 : index} : () -> index
  %1 = "arith.constant"() {"value" = 1000 : index} : () -> index
  %2 = "arith.constant"() {"value" = 3 : index} : () -> index
  %3 = "arith.constant"() {"value" = 3 : i32} : () -> i32
  %4 = "scf.parallel"(%0, %1, %2, %3) ({
    ^bb0(%i: index, %j: i32):
      "scf.yield"() : () -> ()
  }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 1>} : (index, index, index, i32) -> (i32)
}) : () -> ()

// CHECK: scf.yield contains 0 operands but 1 expected
