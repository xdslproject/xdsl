// RUN: xdsl-opt "%s" --verify-diagnostics | filecheck "%s"

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 0 : index} : () -> index
  %1 = "arith.constant"() {"value" = 1000 : index} : () -> index
  %2 = "arith.constant"() {"value" = 3 : index} : () -> index
  "scf.parallel"(%0, %1, %2) ({
  ^bb0(%i: index, %j: index):
    "scf.yield"() : () -> ()
  ^bb1():
    "scf.yield"() : () -> ()
  }) {"operandSegmentSizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
}) : () -> ()

// CHECK:      Operation does not verify: region at position 0 does not verify:
// CHECK-NEXT: expected a single block, but got 2 blocks
