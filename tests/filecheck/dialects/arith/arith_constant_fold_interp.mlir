// RUN: xdsl-opt %s -p constant-fold-interp | filecheck %s

// CHECK: builtin.module {

%i1 = arith.constant 1 : i32
%i2 = arith.constant 2 : i32
%i3 = arith.constant 3 : i32
// CHECK-NEXT: %i1 = arith.constant 1 : i32
// CHECK-NEXT: %i2 = arith.constant 2 : i32
// CHECK-NEXT: %i3 = arith.constant 3 : i32

%sum3 = arith.addi %i1, %i2 : i32
// CHECK-NEXT: %sum3 = arith.constant 3 : i32

%sum6 = arith.addi %sum3, %i3 : i32
// CHECK-NEXT: %sum6 = arith.constant 6 : i32
