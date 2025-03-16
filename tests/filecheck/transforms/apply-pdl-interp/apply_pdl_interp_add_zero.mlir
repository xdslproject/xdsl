// RUN: xdsl-opt %s -p apply-pdl | filecheck %s


// CHECK:       func.func @impl() -> i32 {
// CHECK-NEXT:    %0 = arith.constant 4 : i32
// CHECK-NEXT:    %1 = arith.constant 0 : i32
// CHECK-NEXT:    func.return %0 : i32
// CHECK-NEXT:  }

func.func @impl() -> i32 {
  %0 = arith.constant 4 : i32
  %1 = arith.constant 0 : i32
  %2 = arith.addi %0, %1 : i32
  func.return %2 : i32
}

  pdl_interp.func @matcher(%arg0: !pdl.operation) {
    %0 = pdl_interp.get_operand 1 of %arg0
    %1 = pdl_interp.get_defining_op of %0 : !pdl.value
    pdl_interp.is_not_null %1 : !pdl.operation -> ^bb2, ^bb1
  ^bb1:  // 16 preds: ^bb0, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8, ^bb9, ^bb10, ^bb11, ^bb12, ^bb13, ^bb14, ^bb15, ^bb16
    pdl_interp.finalize
  ^bb2:  // pred: ^bb0
    pdl_interp.check_operation_name of %arg0 is "arith.addi" -> ^bb3, ^bb1
  ^bb3:  // pred: ^bb2
    pdl_interp.check_operand_count of %arg0 is 2 -> ^bb4, ^bb1
  ^bb4:  // pred: ^bb3
    pdl_interp.check_result_count of %arg0 is 1 -> ^bb5, ^bb1
  ^bb5:  // pred: ^bb4
    %2 = pdl_interp.get_operand 0 of %arg0
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb6, ^bb1
  ^bb6:  // pred: ^bb5
    pdl_interp.is_not_null %0 : !pdl.value -> ^bb7, ^bb1
  ^bb7:  // pred: ^bb6
    %3 = pdl_interp.get_result 0 of %arg0
    pdl_interp.is_not_null %3 : !pdl.value -> ^bb8, ^bb1
  ^bb8:  // pred: ^bb7
    pdl_interp.check_operation_name of %1 is "arith.constant" -> ^bb9, ^bb1
  ^bb9:  // pred: ^bb8
    pdl_interp.check_operand_count of %1 is 0 -> ^bb10, ^bb1
  ^bb10:  // pred: ^bb9
    pdl_interp.check_result_count of %1 is 1 -> ^bb11, ^bb1
  ^bb11:  // pred: ^bb10
    %4 = pdl_interp.get_attribute "value" of %1
    pdl_interp.is_not_null %4 : !pdl.attribute -> ^bb12, ^bb1
  ^bb12:  // pred: ^bb11
    pdl_interp.check_attribute %4 is 0 : i32 -> ^bb13, ^bb1
  ^bb13:  // pred: ^bb12
    %5 = pdl_interp.get_result 0 of %1
    pdl_interp.is_not_null %5 : !pdl.value -> ^bb14, ^bb1
  ^bb14:  // pred: ^bb13
    pdl_interp.are_equal %5, %0 : !pdl.value -> ^bb15, ^bb1
  ^bb15:  // pred: ^bb14
    %6 = pdl_interp.get_value_type of %5 : !pdl.type
    %7 = pdl_interp.get_value_type of %3 : !pdl.type
    pdl_interp.are_equal %6, %7 : !pdl.type -> ^bb16, ^bb1
  ^bb16:  // pred: ^bb15
    pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%2, %arg0 : !pdl.value, !pdl.operation) : benefit(2), loc([%1, %arg0]), root("arith.addi") -> ^bb1
  }
  module @rewriters {
    pdl_interp.func @pdl_generated_rewriter(%arg0: !pdl.value, %arg1: !pdl.operation) {
      pdl_interp.replace %arg1 with (%arg0 : !pdl.value)
      pdl_interp.finalize
    }
  }
