// RUN: xdsl-opt %s -p 'apply-individual-rewrite{matched_operation_index=2 operation_name="arith.addi" pattern_name="AdditionOfSameVariablesToMultiplyByTwo"}' | filecheck %s

builtin.module {
  func.func @hello(%n : i32) -> i32 {
    %a = arith.addi %n, %n : i32
    %two = arith.constant 2 : i32
    %res = arith.divui %a, %two : i32
    func.return %res : i32
  }
}


//CHECK:         builtin.module {
// CHECK-NEXT:     func.func @hello(%n : i32) -> i32 {
// CHECK-NEXT:       %a = arith.constant 2 : i32
// CHECK-NEXT:       %a_1 = arith.muli %n, %a : i32
// CHECK-NEXT:       %two = arith.constant 2 : i32
// CHECK-NEXT:       %res = arith.divui %a_1, %two : i32
// CHECK-NEXT:       func.return %res : i32
// CHECK-NEXT:     }
// CHECK-NEXT:  }
