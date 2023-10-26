// RUN: xdsl-opt %s | xdsl-opt | mlir-opt | filecheck %s

%lb = arith.constant 0 : i32
%ub = arith.constant 42 : i32
%step = arith.constant 7 : i32
%sum_init = arith.constant 36 : i32
%sum = scf.for %iv = %lb to %ub step %step iter_args(%sum_iter = %sum_init) -> (i32) : i32 {
  %sum_new = arith.addi %sum_iter, %iv : i32
  scf.yield %sum_new : i32
}

scf.for %iv = %lb to %ub step %step : i32 {
}

// CHECK:      module {
// CHECK-NEXT:   %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 42 : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 7 : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 36 : i32
// CHECK-NEXT:   %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32) : i32 {
// CHECK-NEXT:     %{{.*}} = arith.addi %{{.*}}, %{{.*}}  : i32
// CHECK-NEXT:     scf.yield %{{.*}} : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
// CHECK-NEXT:   }
// CHECK-NEXT: }
