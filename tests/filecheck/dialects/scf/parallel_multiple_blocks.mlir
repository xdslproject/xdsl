// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 0 : index} : () -> index
  %1 = "arith.constant"() {"value" = 1000 : index} : () -> index
  %2 = "arith.constant"() {"value" = 3 : index} : () -> index
  "scf.parallel"(%0, %1, %2) ({
  ^bb0(%i: index, %j: index):
    "scf.reduce"() : () -> ()
  ^bb1():
    "scf.reduce"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
}) : () -> ()

// CHECK:      Operation does not verify: Region 'body' at position 0 expected a single block, but got 2 blocks
