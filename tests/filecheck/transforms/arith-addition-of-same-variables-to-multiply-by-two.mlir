// RUN: xdsl-opt %s -p canonicalize | filecheck %s

func.func @hello(%n : i32) -> i32 {
  %one = arith.constant 2 : i32
  %res = arith.addi %one, %one : i32
  func.return %res : i32
}


//CHECK:         builtin.module {
// CHECK-NEXT:     func.func @hello(%n : i32) -> i32 {
// CHECK-NEXT:       %one = arith.constant 2 : i32
// CHECK-NEXT:       %res = arith.muli %one, 2 : i32
// CHECK-NEXT:       func.return %n : i32
// CHECK-NEXT:     }
// CHECK-NEXT:  }


