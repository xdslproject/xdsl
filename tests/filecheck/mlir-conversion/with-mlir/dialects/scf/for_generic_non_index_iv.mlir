// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  %lb = "arith.constant"() {"value" = 0 : i32} : () -> i32
  %ub = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %step = "arith.constant"() {"value" = 7 : i32} : () -> i32
  %sum_init = "arith.constant"() {"value" = 36 : i32} : () -> i32
  %sum = "scf.for"(%lb, %ub, %step, %sum_init) ({
  ^bb0(%iv : i32, %sum_iter : i32):
    %sum_new = "arith.addi"(%sum_iter, %iv) : (i32, i32) -> i32
    "scf.yield"(%sum_new) : (i32) -> ()
  }) : (i32, i32, i32, i32) -> i32
  "scf.for"(%lb, %ub, %step) ({
  ^bb0(%iv: i32):
    "scf.yield"() : () -> ()
  }) : (i32, i32, i32) -> ()
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %{{.*}} = "arith.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:   %{{.*}} = "arith.constant"() <{value = 42 : i32}> : () -> i32
// CHECK-NEXT:   %{{.*}} = "arith.constant"() <{value = 7 : i32}> : () -> i32
// CHECK-NEXT:   %{{.*}} = "arith.constant"() <{value = 36 : i32}> : () -> i32
// CHECK-NEXT:   %{{.*}} = "scf.for"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ({
// CHECK-NEXT:   ^bb0(%{{.*}} : i32, %{{.*}} : i32):
// CHECK-NEXT:     %{{.*}} = "arith.addi"(%{{.*}}, %{{.*}}) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK-NEXT:     "scf.yield"(%{{.*}}) : (i32) -> ()
// CHECK-NEXT:   }) : (i32, i32, i32, i32) -> i32
// CHECK-NEXT:   "scf.for"(%{{.*}}, %{{.*}}, %{{.*}}) ({
// CHECK-NEXT:   ^bb1(%{{.*}}: i32):
// CHECK-NEXT:     "scf.yield"() : () -> ()
// CHECK-NEXT:   }) : (i32, i32, i32) -> ()
// CHECK-NEXT: }) : () -> ()
