// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 0 : index} : () -> index
  %1 = "arith.constant"() {"value" = 1000 : index} : () -> index
  %2 = "arith.constant"() {"value" = 3 : index} : () -> index
  "scf.parallel"(%0, %1, %2) ({
  ^bb0(%arg0: index):
    "scf.yield"() : () -> ()
  }) {"operandSegmentSizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
}) : () -> ()

// CHECK-NEXT: "builtin.module"() ({
// CHECK-NEXT:   %{{.*}} = "arith.constant"() <{"value" = 0 : index}> : () -> index
// CHECK-NEXT:   %{{.*}} = "arith.constant"() <{"value" = 1000 : index}> : () -> index
// CHECK-NEXT:   %{{.*}} = "arith.constant"() <{"value" = 3 : index}> : () -> index
// CHECK-NEXT:   "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:   ^{{.*}}(%{{.*}}: index):
// CHECK-NEXT:     "scf.yield"() : () -> ()
// CHECK-NEXT:   }) : (index, index, index) -> ()
// CHECK-NEXT: }) : () -> ()
