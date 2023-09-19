// RUN: xdsl-opt %s -p constant-fold-interp | filecheck %s

// CHECK: builtin.module {

%i1 = arith.constant 1 : i32
%i2 = arith.constant 2 : i32
// CHECK-NEXT: %i1 = arith.constant 1 : i32
// CHECK-NEXT: %i2 = arith.constant 2 : i32

%addi = arith.addi %i1, %i2 : i32
// CHECK-NEXT: %addi = arith.constant 3 : i32

