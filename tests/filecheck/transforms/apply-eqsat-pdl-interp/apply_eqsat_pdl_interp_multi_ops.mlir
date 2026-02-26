// RUN: xdsl-opt %s -p apply-eqsat-pdl-interp | filecheck %s

// CHECK:      func.func @impl() -> i32 {
// CHECK-NEXT:   %a = arith.constant 3 : i32
// CHECK-NEXT:   %a_1 = equivalence.class %a : i32
// CHECK-NEXT:   %b = arith.constant 5 : i32
// CHECK-NEXT:   %b_1 = equivalence.class %b : i32
// CHECK-NEXT:   %c = arith.constant 7 : i32
// CHECK-NEXT:   %c_1 = equivalence.class %c : i32
// CHECK-NEXT:   %sum = arith.constant 8 : i32
// CHECK-NEXT:   %d = arith.addi %a_1, %b_1 : i32
// CHECK-NEXT:   %d_1 = equivalence.class %sum, %d : i32
// CHECK-NEXT:   %0 = arith.subi %b_1, %c_1 : i32
// CHECK-NEXT:   %1 = equivalence.class %0 : i32
// CHECK-NEXT:   %2 = arith.addi %a_1, %1 : i32
// CHECK-NEXT:   %e = arith.subi %d_1, %c_1 : i32
// CHECK-NEXT:   %e_1 = equivalence.class %e, %2 : i32
// CHECK-NEXT:   func.return %e_1 : i32
// CHECK-NEXT: }


func.func @impl() -> i32 {
  %a = arith.constant 3 : i32
  %a_1 = equivalence.class %a : i32
  %b = arith.constant 5 : i32
  %b_1 = equivalence.class %b : i32
  %c = arith.constant 7 : i32
  %c_1 = equivalence.class %c : i32
  %sum = arith.constant 8 : i32
  %d = arith.addi %a_1, %b_1 : i32
  %d_1 = equivalence.class %sum, %d : i32
  %e = arith.subi %d_1, %c_1 : i32
  %e_1 = equivalence.class %e : i32
  func.return %e_1 : i32
}

pdl_interp.func @matcher(%arg0: !pdl.operation) {
  %0 = eqsat_pdl_interp.get_result 0 of %arg0
  pdl_interp.is_not_null %0 : !pdl.value -> ^bb2, ^bb1
^bb1:  // 16 preds: ^bb0, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8, ^bb9, ^bb10, ^bb11, ^bb12, ^bb13, ^bb14, ^bb15, ^bb16
  eqsat_pdl_interp.finalize
^bb2:  // pred: ^bb0
  %1 = pdl_interp.get_operand 0 of %arg0
  %2 = eqsat_pdl_interp.get_defining_op of %1 : !pdl.value
  pdl_interp.is_not_null %2 : !pdl.operation -> ^bb3, ^bb1
^bb3:  // pred: ^bb2
  pdl_interp.check_operation_name of %arg0 is "arith.subi" -> ^bb4, ^bb1
^bb4:  // pred: ^bb3
  pdl_interp.check_operand_count of %arg0 is 2 -> ^bb5, ^bb1
^bb5:  // pred: ^bb4
  pdl_interp.check_result_count of %arg0 is 1 -> ^bb6, ^bb1
^bb6:  // pred: ^bb5
  pdl_interp.is_not_null %1 : !pdl.value -> ^bb7, ^bb1
^bb7:  // pred: ^bb6
  %3 = pdl_interp.get_operand 1 of %arg0
  pdl_interp.is_not_null %3 : !pdl.value -> ^bb8, ^bb1
^bb8:  // pred: ^bb7
  pdl_interp.check_operation_name of %2 is "arith.addi" -> ^bb9, ^bb1
^bb9:  // pred: ^bb8
  pdl_interp.check_operand_count of %2 is 2 -> ^bb10, ^bb1
^bb10:  // pred: ^bb9
  pdl_interp.check_result_count of %2 is 1 -> ^bb11, ^bb1
^bb11:  // pred: ^bb10
  %4 = pdl_interp.get_operand 0 of %2
  pdl_interp.is_not_null %4 : !pdl.value -> ^bb12, ^bb1
^bb12:  // pred: ^bb11
  %5 = pdl_interp.get_operand 1 of %2
  pdl_interp.is_not_null %5 : !pdl.value -> ^bb13, ^bb1
^bb13:  // pred: ^bb12
  %6 = eqsat_pdl_interp.get_result 0 of %2
  pdl_interp.is_not_null %6 : !pdl.value -> ^bb14, ^bb1
^bb14:  // pred: ^bb13
  pdl_interp.are_equal %6, %1 : !pdl.value -> ^bb15, ^bb1
^bb15:  // pred: ^bb14
  %7 = pdl_interp.get_value_type of %6 : !pdl.type
  %8 = pdl_interp.get_value_type of %0 : !pdl.type
  pdl_interp.are_equal %7, %8 : !pdl.type -> ^bb16, ^bb1
^bb16:  // pred: ^bb15
  eqsat_pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%5, %3, %7, %4, %arg0 : !pdl.value, !pdl.value, !pdl.type, !pdl.value, !pdl.operation) : benefit(1), generatedOps(["arith.subi", "arith.addi"]), loc([%2, %arg0]), root("arith.subi") -> ^bb1
}
module @rewriters {
  pdl_interp.func @pdl_generated_rewriter(%arg0: !pdl.value, %arg1: !pdl.value, %arg2: !pdl.type, %arg3: !pdl.value, %arg4: !pdl.operation) {
    %0 = eqsat_pdl_interp.create_operation "arith.subi"(%arg0, %arg1 : !pdl.value, !pdl.value)  -> (%arg2 : !pdl.type)
    %1 = eqsat_pdl_interp.get_result 0 of %0
    %2 = eqsat_pdl_interp.create_operation "arith.addi"(%arg3, %1 : !pdl.value, !pdl.value)  -> (%arg2 : !pdl.type)
    %3 = eqsat_pdl_interp.get_result 0 of %2
    %4 = eqsat_pdl_interp.get_results of %2 : !pdl.range<value>
    eqsat_pdl_interp.replace %arg4 with (%4 : !pdl.range<value>)
    eqsat_pdl_interp.finalize
  }
}
