// RUN: xdsl-opt %s --print-op-generic --split-input-file | filecheck %s

"builtin.module"() ({
  %lb = "arith.constant"() {"value" = 0 : index} : () -> index
  %ub = "arith.constant"() {"value" = 42 : index} : () -> index
  %step = "arith.constant"() {"value" = 7 : index} : () -> index
  "scf.for"(%lb, %ub, %step) ({
  ^0(%iv : index):
  }) : (index, index, index) -> ()
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:  %lb = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:  %ub = "arith.constant"() {"value" = 42 : index} : () -> index
// CHECK-NEXT:  %step = "arith.constant"() {"value" = 7 : index} : () -> index
// CHECK-NEXT:  "scf.for"(%lb, %ub, %step) ({
// CHECK-NEXT:  ^0(%iv : index):
// CHECK-NEXT:    "scf.yield"() : () -> ()
// CHECK-NEXT:  }) : (index, index, index) -> ()
// CHECK-NEXT:}) : () -> ()

// -----

"builtin.module"() ({
  %lb = "arith.constant"() {"value" = 0 : index} : () -> index
  %ub = "arith.constant"() {"value" = 42 : index} : () -> index
  %step = "arith.constant"() {"value" = 7 : index} : () -> index
  "scf.for"(%lb, %ub, %step) ({
  ^0(%iv : index):
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:  %lb = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:  %ub = "arith.constant"() {"value" = 42 : index} : () -> index
// CHECK-NEXT:  %step = "arith.constant"() {"value" = 7 : index} : () -> index
// CHECK-NEXT:  "scf.for"(%lb, %ub, %step) ({
// CHECK-NEXT:  ^0(%iv : index):
// CHECK-NEXT:    "scf.yield"() : () -> ()
// CHECK-NEXT:  }) : (index, index, index) -> ()
// CHECK-NEXT:}) : () -> ()
