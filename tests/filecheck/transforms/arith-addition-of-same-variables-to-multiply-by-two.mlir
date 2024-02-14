// RUN: xdsl-opt %s -p canonicalize | filecheck %s

func.func @hello(%n : i32) -> i32 {
  %two = arith.constant 3 : i32
  %res = arith.addi %two, %two : i32
  func.return %res : i32
}

//CHECK:         builtin.module {
// CHECK-NEXT:     func.func @hello(%n : i32) -> i32 {
// CHECK-NEXT:       %two = arith.constant 3 : i32
// CHECK-NEXT:       %res = arith.constant 2 : i32
// CHECK-NEXT:       %res_1 = arith.muli %two, %res : i32
// CHECK-NEXT:       func.return %res_1 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:  }


