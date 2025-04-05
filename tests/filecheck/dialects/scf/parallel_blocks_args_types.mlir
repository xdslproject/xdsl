// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 0 : index} : () -> index
  %1 = "arith.constant"() {"value" = 1000 : index} : () -> index
  %2 = "arith.constant"() {"value" = 3 : index} : () -> index
  "scf.parallel"(%0, %1, %2) ({
    ^bb0(%i: i32):
      "scf.reduce"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
}) : () -> ()

// CHECK: scf.parallel's block must have an index argument for each induction variable
