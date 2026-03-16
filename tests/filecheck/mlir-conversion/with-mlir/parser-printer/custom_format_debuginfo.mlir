// RUN: xdsl-opt %s --print-debuginfo | mlir-opt --allow-unregistered-dialect --mlir-print-debuginfo --mlir-print-local-scope | xdsl-opt --print-debuginfo | filecheck %s --check-prefix=CHECK-DEBUG-INFO

%0 = arith.constant 1 : i32
%1 = arith.constant 2 : i32
%2 = arith.addi %0, %1 : i32 loc("model.mlir":7:9)

// CHECK-DEBUG-INFO: builtin.module {
// CHECK-DEBUG-INFO-NEXT:   %{{.*}} = arith.constant 1 : i32 loc(unknown)
// CHECK-DEBUG-INFO-NEXT:   %{{.*}} = arith.constant 2 : i32 loc(unknown)
// CHECK-DEBUG-INFO-NEXT:   %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32 loc("model.mlir":7:9)
// CHECK-DEBUG-INFO-NEXT: } loc(unknown)
