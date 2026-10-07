// RUN: xdsl-opt %s | xdsl-opt | $XDSL_MLIR_OPT | filecheck %s

%lb = arith.constant 0 : index
%ub = arith.constant 42 : index
%step = arith.constant 7 : index
%sum_init = arith.constant 36 : index
%sum = scf.for %iv = %lb to %ub step %step iter_args(%sum_iter = %sum_init) -> (index) {
  %sum_new = arith.addi %sum_iter, %iv : index
  scf.yield %sum_new : index
}

scf.for %iv = %lb to %ub step %step {
}

%sum_attr = scf.for %iv = %lb to %ub step %step iter_args(%sum_iter = %sum_init) -> (index) {
  scf.yield %sum_iter : index
} {hello = "world"}

scf.for %iv = %lb to %ub step %step {
} {hello = "world"}

// CHECK:      module {
// CHECK-NEXT:   %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:   %{{.*}} = arith.constant 42 : index
// CHECK-NEXT:   %{{.*}} = arith.constant 7 : index
// CHECK-NEXT:   %{{.*}} = arith.constant 36 : index
// CHECK-NEXT:   %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index) {
// CHECK-NEXT:     %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     scf.yield %{{.*}} : index
// CHECK-NEXT:   }
// CHECK-NEXT:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:   }
// CHECK-NEXT:   %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index) {
// CHECK-NEXT:     scf.yield %{{.*}} : index
// CHECK-NEXT:   } {hello = "world"}
// CHECK-NEXT:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:   } {hello = "world"}
// CHECK-NEXT: }
