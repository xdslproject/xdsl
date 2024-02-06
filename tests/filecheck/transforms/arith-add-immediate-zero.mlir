// RUN: xdsl-opt %s -p 'apply-individual-rewrite{matched_operation_index=5 operation_name="arith.addi" pattern_name="AddImmediateZero"}'| filecheck %s

func.func @hello(%n : i32) -> i32 {
  %two = arith.constant 0 : i32
  %three = arith.constant 0 : i32
  %res = arith.addi %two, %n : i32
  %res2 = arith.addi %three, %res : i32
  func.return %res : i32
}


//CHECK:         builtin.module {
// CHECK-NEXT:     func.func @hello(%n : i32) -> i32 {
// CHECK-NEXT:       %two = arith.constant 0 : i32
// CHECK-NEXT:       %three = arith.constant 0 : i32
// CHECK-NEXT:       %res = arith.addi %two, %n : i32
// CHECK-NEXT:       func.return %res : i32
// CHECK-NEXT:     }
// CHECK-NEXT:  }


