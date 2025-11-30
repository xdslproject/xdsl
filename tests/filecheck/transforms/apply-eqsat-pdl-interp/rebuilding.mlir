// RUN: xdsl-opt %s -p apply-eqsat-pdl-interp | filecheck %s

// CHECK:      %x_c = eqsat.eclass %x : i32
// CHECK-NEXT: %zero = arith.constant 0 : i32
// CHECK-NEXT: %a = arith.muli %x_c, %a_c : i32
// CHECK-NEXT: %a_c = eqsat.eclass %a, %zero, %b : i32
// CHECK-NEXT: %b = arith.subi %x_c, %x_c : i32
// CHECK-NEXT: func.return %a_c, %a_c : i32, i32

func.func @impl(%x: i32) -> (i32, i32) {
  %x_c = eqsat.eclass %x : i32

  %zero = arith.constant 0 : i32
  %zero_c = eqsat.eclass %zero : i32

  %a = arith.muli %x_c, %zero_c : i32
  %a_c = eqsat.eclass %a : i32

  %b = arith.subi %x_c, %x_c : i32
  %b_c = eqsat.eclass %b : i32

  
  func.return %a_c, %b_c : i32, i32
}

pdl_interp.func @matcher(%arg0: !pdl.operation) {
  pdl_interp.switch_operation_name of %arg0 to ["arith.muli", "arith.subi"](^bb2, ^bb17) -> ^bb1
^bb1:  // 22 preds: ^bb0, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8, ^bb9, ^bb10, ^bb11, ^bb12, ^bb13, ^bb14, ^bb15, ^bb16, ^bb17, ^bb18, ^bb19, ^bb20, ^bb21, ^bb22
  eqsat_pdl_interp.finalize
^bb2:  // pred: ^bb0
  pdl_interp.check_operand_count of %arg0 is 2 -> ^bb3, ^bb1
^bb3:  // pred: ^bb2
  pdl_interp.check_result_count of %arg0 is 1 -> ^bb4, ^bb1
^bb4:  // pred: ^bb3
  %0 = pdl_interp.get_operand 0 of %arg0
  pdl_interp.is_not_null %0 : !pdl.value -> ^bb5, ^bb1
^bb5:  // pred: ^bb4
  %1 = eqsat_pdl_interp.get_result 0 of %arg0
  pdl_interp.is_not_null %1 : !pdl.value -> ^bb6, ^bb1
^bb6:  // pred: ^bb5
  %2 = pdl_interp.get_operand 1 of %arg0
  %3 = eqsat_pdl_interp.get_defining_op of %2 : !pdl.value
  pdl_interp.is_not_null %3 : !pdl.operation -> ^bb7, ^bb1
^bb7:  // pred: ^bb6
  pdl_interp.is_not_null %2 : !pdl.value -> ^bb8, ^bb1
^bb8:  // pred: ^bb7
  pdl_interp.check_operation_name of %3 is "arith.constant" -> ^bb9, ^bb1
^bb9:  // pred: ^bb8
  pdl_interp.check_operand_count of %3 is 0 -> ^bb10, ^bb1
^bb10:  // pred: ^bb9
  pdl_interp.check_result_count of %3 is 1 -> ^bb11, ^bb1
^bb11:  // pred: ^bb10
  %4 = pdl_interp.get_attribute "value" of %3
  pdl_interp.is_not_null %4 : !pdl.attribute -> ^bb12, ^bb1
^bb12:  // pred: ^bb11
  pdl_interp.check_attribute %4 is 0 : i32 -> ^bb13, ^bb1
^bb13:  // pred: ^bb12
  %5 = eqsat_pdl_interp.get_result 0 of %3
  pdl_interp.is_not_null %5 : !pdl.value -> ^bb14, ^bb1
^bb14:  // pred: ^bb13
  pdl_interp.are_equal %5, %2 : !pdl.value -> ^bb15, ^bb1
^bb15:  // pred: ^bb14
  %6 = pdl_interp.get_value_type of %5 : !pdl.type
  %7 = pdl_interp.get_value_type of %1 : !pdl.type
  pdl_interp.are_equal %6, %7 : !pdl.type -> ^bb16, ^bb1
^bb16:  // pred: ^bb15
  eqsat_pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%3, %arg0 : !pdl.operation, !pdl.operation) : benefit(1), loc([%arg0, %3]), root("arith.muli") -> ^bb1
^bb17:  // pred: ^bb0
  pdl_interp.check_operand_count of %arg0 is 2 -> ^bb18, ^bb1
^bb18:  // pred: ^bb17
  pdl_interp.check_result_count of %arg0 is 1 -> ^bb19, ^bb1
^bb19:  // pred: ^bb18
  %8 = pdl_interp.get_operand 0 of %arg0
  pdl_interp.is_not_null %8 : !pdl.value -> ^bb20, ^bb1
^bb20:  // pred: ^bb19
  %9 = eqsat_pdl_interp.get_result 0 of %arg0
  pdl_interp.is_not_null %9 : !pdl.value -> ^bb21, ^bb1
^bb21:  // pred: ^bb20
  %10 = pdl_interp.get_operand 1 of %arg0
  pdl_interp.are_equal %8, %10 : !pdl.value -> ^bb22, ^bb1
^bb22:  // pred: ^bb21
  %11 = pdl_interp.get_value_type of %9 : !pdl.type
  eqsat_pdl_interp.record_match @rewriters::@pdl_generated_rewriter_0(%11, %arg0 : !pdl.type, !pdl.operation) : benefit(1), generatedOps(["arith.constant"]), loc([%arg0]), root("arith.subi") -> ^bb1
}
module @rewriters {
  pdl_interp.func @pdl_generated_rewriter(%arg0: !pdl.operation, %arg1: !pdl.operation) {
    %0 = eqsat_pdl_interp.get_results of %arg0 : !pdl.range<value>
    eqsat_pdl_interp.replace %arg1 with (%0 : !pdl.range<value>)
    eqsat_pdl_interp.finalize
  }
  pdl_interp.func @pdl_generated_rewriter_0(%arg0: !pdl.type, %arg1: !pdl.operation) {
    %0 = pdl_interp.create_attribute 0 : i32
    %1 = eqsat_pdl_interp.create_operation "arith.constant" {"value" = %0}  -> (%arg0 : !pdl.type)
    %2 = eqsat_pdl_interp.get_results of %1 : !pdl.range<value>
    eqsat_pdl_interp.replace %arg1 with (%2 : !pdl.range<value>)
    eqsat_pdl_interp.finalize
  }
}

// // x * 0 -> 0
// pdl.pattern : benefit(1) {
//   %x = pdl.operand
//   %type = pdl.type
//   %zero = pdl.attribute = 0 : i32
//   %constop = pdl.operation "arith.constant" {"value" = %zero} -> (%type : !pdl.type)
//   %const = pdl.result 0 of %constop
//   %mulop = pdl.operation "arith.muli" (%x, %const : !pdl.value, !pdl.value) -> (%type : !pdl.type)
//   pdl.rewrite %mulop {
//     pdl.replace %mulop with %constop
//   }
// }

// // x - x -> 0
// pdl.pattern : benefit(1) {
//   %x = pdl.operand
//   %type = pdl.type
//   %subop = pdl.operation "arith.subi" (%x, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
//   pdl.rewrite %subop {
//     %zero = pdl.attribute = 0 : i32
//     %constop = pdl.operation "arith.constant" {"value" = %zero} -> (%type : !pdl.type)
//     pdl.replace %subop with %constop
//   }
// }
