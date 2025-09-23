// RUN: xdsl-opt %s -p test-constant-folding --split-input-file | filecheck %s

builtin.module {

  // CHECK:      builtin.module {
  // CHECK-NEXT:   %0 = arith.constant 865 : i32
  // CHECK-NEXT:   %1 = arith.constant 395 : i32
  // CHECK-NEXT:   %2 = arith.constant 1260 : i32
  // CHECK-NEXT:   %3 = arith.constant 777 : i32
  // CHECK-NEXT:   %4 = arith.constant 2037 : i32
  // CHECK-NEXT:   %5 = arith.constant 912 : i32
  // CHECK-NEXT:   %6 = arith.constant 2949 : i32
  // CHECK-NEXT:   %7 = arith.constant 431 : i32
  // CHECK-NEXT:   %8 = arith.constant 3380 : i32
  // CHECK-NEXT:   %9 = arith.constant 42 : i32
  // CHECK-NEXT:   %10 = arith.constant 3422 : i32
  // CHECK-NEXT:   "test.op"(%10) : (i32) -> ()
  // CHECK-NEXT: }

  %0 = arith.constant 865 : i32
  %1 = arith.constant 395 : i32
  %2 = arith.addi %1, %0 : i32
  %3 = arith.constant 777 : i32
  %4 = arith.addi %3, %2 : i32
  %5 = arith.constant 912 : i32
  %6 = arith.addi %5, %4 : i32
  %7 = arith.constant 431 : i32
  %8 = arith.addi %7, %6 : i32
  %9 = arith.constant 42 : i32
  %10 = arith.addi %9, %8 : i32
  "test.op"(%10) : (i32) -> ()
}
