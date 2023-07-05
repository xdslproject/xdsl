// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  %lb = "arith.constant"() {"value" = 0 : index} : () -> index
  %ub = "arith.constant"() {"value" = 42 : index} : () -> index
  %step = "arith.constant"() {"value" = 7 : index} : () -> index
  %sum_init = "arith.constant"() {"value" = 36 : index} : () -> index
  %sum = "scf.for"(%lb, %ub, %step, %sum_init) ({
  ^0(%iv : index, %sum_iter : index):
    %sum_new = "arith.addi"(%sum_iter, %iv) : (index, index) -> index
    "scf.yield"(%sum_new) : (index) -> ()
  }) : (index, index, index, index) -> index
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %{{.*}} = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:   %{{.*}} = "arith.constant"() {"value" = 42 : index} : () -> index
// CHECK-NEXT:   %{{.*}} = "arith.constant"() {"value" = 7 : index} : () -> index
// CHECK-NEXT:   %{{.*}} = "arith.constant"() {"value" = 36 : index} : () -> index
// CHECK-NEXT:   %{{.*}} = "scf.for"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ({
// CHECK-NEXT:   ^0(%{{.*}} : index, %{{.*}} : index):
// CHECK-NEXT:     %{{.*}} = "arith.addi"(%{{.*}}, %{{.*}}) : (index, index) -> index
// CHECK-NEXT:     "scf.yield"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   }) : (index, index, index, index) -> index
// CHECK-NEXT: }) : () -> ()
