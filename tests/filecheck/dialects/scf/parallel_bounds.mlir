// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 0 : index} : () -> index
  %1 = "arith.constant"() {"value" = 1000 : index} : () -> index
  %2 = "arith.constant"() {"value" = 3 : index} : () -> index
  %3 = "arith.constant"() {"value" = 0 : index} : () -> index
  %4 = "arith.constant"() {"value" = 1000 : index} : () -> index
  "scf.parallel"(%0, %3, %1, %4, %2) ({
  ^bb0(%i: index):
    "scf.reduce"() : () -> ()
  }) {operandSegmentSizes = array<i32: 2, 2, 1, 0>} : (index, index, index, index, index) -> ()
}) : () -> ()

// CHECK: Expected the same number of lower bounds, upper bounds, and steps for scf.parallel. Got 2, 2 and 1.
