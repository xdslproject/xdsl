// RUN: xdsl-opt %s -p apply-pdl-interp | filecheck %s


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

"pdl_interp.func"() <{function_type = (!pdl.operation) -> (), sym_name = "matcher"}> ({
^bb0(%arg2: !pdl.operation):
  %0 = "pdl_interp.get_operand"(%arg2) <{index = 1 : i32}> : (!pdl.operation) -> !pdl.value
  %1 = "pdl_interp.get_defining_op"(%0) : (!pdl.value) -> !pdl.operation
  "pdl_interp.is_not_null"(%1)[^bb2, ^bb1] : (!pdl.operation) -> ()
^bb1:  // 16 preds: ^bb0, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8, ^bb9, ^bb10, ^bb11, ^bb12, ^bb13, ^bb14, ^bb15, ^bb16
  "pdl_interp.finalize"() : () -> ()
^bb2:  // pred: ^bb0
  "pdl_interp.check_operation_name"(%arg2)[^bb3, ^bb1] <{name = "arith.addi"}> : (!pdl.operation) -> ()
^bb3:  // pred: ^bb2
  "pdl_interp.check_operand_count"(%arg2)[^bb4, ^bb1] <{count = 2 : i32}> : (!pdl.operation) -> ()
^bb4:  // pred: ^bb3
  "pdl_interp.check_result_count"(%arg2)[^bb5, ^bb1] <{count = 1 : i32}> : (!pdl.operation) -> ()
^bb5:  // pred: ^bb4
  %2 = "pdl_interp.get_operand"(%arg2) <{index = 0 : i32}> : (!pdl.operation) -> !pdl.value
  "pdl_interp.is_not_null"(%2)[^bb6, ^bb1] : (!pdl.value) -> ()
^bb6:  // pred: ^bb5
  "pdl_interp.is_not_null"(%0)[^bb7, ^bb1] : (!pdl.value) -> ()
^bb7:  // pred: ^bb6
  %3 = "pdl_interp.get_result"(%arg2) <{index = 0 : i32}> : (!pdl.operation) -> !pdl.value
  "pdl_interp.is_not_null"(%3)[^bb8, ^bb1] : (!pdl.value) -> ()
^bb8:  // pred: ^bb7
  "pdl_interp.check_operation_name"(%1)[^bb9, ^bb1] <{name = "arith.constant"}> : (!pdl.operation) -> ()
^bb9:  // pred: ^bb8
  "pdl_interp.check_operand_count"(%1)[^bb10, ^bb1] <{count = 0 : i32}> : (!pdl.operation) -> ()
^bb10:  // pred: ^bb9
  "pdl_interp.check_result_count"(%1)[^bb11, ^bb1] <{count = 1 : i32}> : (!pdl.operation) -> ()
^bb11:  // pred: ^bb10
  %4 = "pdl_interp.get_attribute"(%1) <{name = "value"}> : (!pdl.operation) -> !pdl.attribute
  "pdl_interp.is_not_null"(%4)[^bb12, ^bb1] : (!pdl.attribute) -> ()
^bb12:  // pred: ^bb11
  "pdl_interp.check_attribute"(%4)[^bb13, ^bb1] <{constantValue = 0 : i32}> : (!pdl.attribute) -> ()
^bb13:  // pred: ^bb12
  %5 = "pdl_interp.get_result"(%1) <{index = 0 : i32}> : (!pdl.operation) -> !pdl.value
  "pdl_interp.is_not_null"(%5)[^bb14, ^bb1] : (!pdl.value) -> ()
^bb14:  // pred: ^bb13
  "pdl_interp.are_equal"(%5, %0)[^bb15, ^bb1] : (!pdl.value, !pdl.value) -> ()
^bb15:  // pred: ^bb14
  %6 = "pdl_interp.get_value_type"(%5) : (!pdl.value) -> !pdl.type
  %7 = "pdl_interp.get_value_type"(%3) : (!pdl.value) -> !pdl.type
  "pdl_interp.are_equal"(%6, %7)[^bb16, ^bb1] : (!pdl.type, !pdl.type) -> ()
^bb16:  // pred: ^bb15
  "pdl_interp.record_match"(%2, %arg2, %1, %arg2)[^bb1] <{benefit = 2 : i16, operandSegmentSizes = array<i32: 2, 2>, rewriter = @rewriters::@pdl_generated_rewriter, rootKind = "arith.addi"}> : (!pdl.value, !pdl.operation, !pdl.operation, !pdl.operation) -> ()
}) : () -> ()

"builtin.module"() <{sym_name = "rewriters"}> ({
  "pdl_interp.func"() <{function_type = (!pdl.value, !pdl.operation) -> (), sym_name = "pdl_generated_rewriter"}> ({
  ^bb0(%arg0: !pdl.value, %arg1: !pdl.operation):
    "pdl_interp.replace"(%arg1, %arg0) : (!pdl.operation, !pdl.value) -> ()
    "pdl_interp.finalize"() : () -> ()
  }) : () -> ()
}) : () -> ()
