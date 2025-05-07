// RUN: xdsl-opt %s -p canonicalize | filecheck %s

func.func @hello(%n : i32) -> i32 {
  %two = arith.constant 0 : i32
  %three = arith.constant 0 : i32
  %res = arith.addi %two, %n : i32
  %res2 = arith.addi %three, %res : i32
  func.return %res : i32
}


//CHECK:         builtin.module {
// CHECK-NEXT:     func.func @hello(%n : i32) -> i32 {
// CHECK-NEXT:       func.return %n : i32
// CHECK-NEXT:     }
// CHECK-NEXT:  }
