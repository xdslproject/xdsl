// RUN: xdsl-opt %s --print-op-generic | mlir-opt --mlir-print-op-generic | xdsl-opt | filecheck %s
// RUN: xdsl-opt %s | mlir-opt | xdsl-opt | filecheck %s

pdl_interp.func @matcher(%arg0: !pdl.operation) {
  %0 = pdl_interp.get_result 0 of %arg0
  pdl_interp.is_not_null %0 : !pdl.value -> ^bb2, ^bb1
^bb1:  // 16 preds: ^bb0, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8, ^bb9, ^bb10, ^bb11, ^bb12, ^bb13, ^bb14, ^bb15, ^bb16
  pdl_interp.finalize
^bb2:  // pred: ^bb0
  %1 = pdl_interp.get_operand 0 of %arg0
  %2 = pdl_interp.get_defining_op of %1 : !pdl.value
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
  %6 = pdl_interp.get_result 0 of %2
  pdl_interp.is_not_null %6 : !pdl.value -> ^bb14, ^bb1
^bb14:  // pred: ^bb13
  pdl_interp.are_equal %6, %1 : !pdl.value -> ^bb15, ^bb1
^bb15:  // pred: ^bb14
  %7 = pdl_interp.get_value_type of %6 : !pdl.type
  %8 = pdl_interp.get_value_type of %0 : !pdl.type
  pdl_interp.are_equal %7, %8 : !pdl.type -> ^bb16, ^bb1
^bb16:  // pred: ^bb15
  pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%5, %3, %7, %4, %arg0 : !pdl.value, !pdl.value, !pdl.type, !pdl.value, !pdl.operation) : benefit(1), generatedOps(["arith.subi", "arith.addi"]), loc([%2, %arg0]), root("arith.subi") -> ^bb1
^bb17:
  pdl_interp.switch_operation_name of %arg0 to ["foo.op", "bar.op"](^bb1, ^bb18) -> ^bb3
^bb18:
  pdl_interp.check_type %7 is i32 -> ^bb16, ^bb1
^bb19:
  %attr_val = pdl_interp.get_attribute "test_attr" of %arg0
  pdl_interp.switch_attribute %attr_val to [42 : i32, true](^bb20, ^bb1) -> ^bb1
^bb20:
  %9 = pdl_interp.apply_constraint "myConstraint"(%attr_val : !pdl.attribute) : !pdl.operation {isNegated = true} -> ^bb16, ^bb1
^bb21:
  pdl_interp.apply_rewrite "myRewriter"(%arg0 : !pdl.operation)
  pdl_interp.finalize
^bb22:
  %10 = pdl_interp.apply_rewrite "myRewriter"(%arg0 : !pdl.operation) : !pdl.operation
  pdl_interp.finalize
^bb23:
  %11 = pdl_interp.apply_rewrite "myRewriter"(%arg0, %0 : !pdl.operation, !pdl.value) : !pdl.attribute
  pdl_interp.finalize
}
module @rewriters {
  pdl_interp.func @pdl_generated_rewriter(%arg0: !pdl.value, %arg1: !pdl.value, %arg2: !pdl.type, %arg3: !pdl.value, %arg4: !pdl.operation) {
    %attr = pdl_interp.create_attribute 10 : i64 
    %type_i64 = pdl_interp.create_type i64
    %types_range = pdl_interp.create_types [i32, i64]
    %0 = pdl_interp.create_operation "arith.subi"(%arg0, %arg1 : !pdl.value, !pdl.value) {"attrA" = %attr}  -> (%arg2 : !pdl.type)
    %nooperands = pdl_interp.create_operation "test.testop" {"attrA" = %attr} -> (%arg2 : !pdl.type)
    %1 = pdl_interp.get_result 0 of %0
    %2 = pdl_interp.create_operation "arith.addi"(%arg3, %1 : !pdl.value, !pdl.value)  -> (%arg2 : !pdl.type)
    %3 = pdl_interp.get_result 0 of %2
    %4 = pdl_interp.get_results of %2 : !pdl.range<value>
    %5 = pdl_interp.get_results 0 of %2 : !pdl.range<value>
    pdl_interp.replace %arg4 with (%4 : !pdl.range<value>)
    pdl_interp.finalize
  }
}

// CHECK:     builtin.module {
// CHECK-NEXT:     pdl_interp.func @matcher([[arg0:%arg.]] : !pdl.operation) {
// CHECK-NEXT:       %0 = pdl_interp.get_result 0 of [[arg0]]
// CHECK-NEXT:       pdl_interp.is_not_null %0 : !pdl.value -> ^bb0, ^bb1
// CHECK-NEXT:     ^bb1:
// CHECK-NEXT:       pdl_interp.finalize
// CHECK-NEXT:     ^bb0:
// CHECK-NEXT:       %1 = pdl_interp.get_operand 0 of [[arg0]]
// CHECK-NEXT:       %2 = pdl_interp.get_defining_op of %1 : !pdl.value
// CHECK-NEXT:       pdl_interp.is_not_null %2 : !pdl.operation -> ^bb2, ^bb1
// CHECK-NEXT:     ^bb2:
// CHECK-NEXT:       pdl_interp.check_operation_name of [[arg0]] is "arith.subi" -> ^bb3, ^bb1
// CHECK-NEXT:     ^bb3:
// CHECK-NEXT:       pdl_interp.check_operand_count of [[arg0]] is 2 -> ^bb4, ^bb1
// CHECK-NEXT:     ^bb4:
// CHECK-NEXT:       pdl_interp.check_result_count of [[arg0]] is 1 -> ^bb5, ^bb1
// CHECK-NEXT:     ^bb5:
// CHECK-NEXT:       pdl_interp.is_not_null %1 : !pdl.value -> ^bb6, ^bb1
// CHECK-NEXT:     ^bb6:
// CHECK-NEXT:       %3 = pdl_interp.get_operand 1 of [[arg0]]
// CHECK-NEXT:       pdl_interp.is_not_null %3 : !pdl.value -> ^bb7, ^bb1
// CHECK-NEXT:     ^bb7:
// CHECK-NEXT:       pdl_interp.check_operation_name of %2 is "arith.addi" -> ^bb8, ^bb1
// CHECK-NEXT:     ^bb8:
// CHECK-NEXT:       pdl_interp.check_operand_count of %2 is 2 -> ^bb9, ^bb1
// CHECK-NEXT:     ^bb9:
// CHECK-NEXT:       pdl_interp.check_result_count of %2 is 1 -> ^bb10, ^bb1
// CHECK-NEXT:     ^bb10:
// CHECK-NEXT:       %4 = pdl_interp.get_operand 0 of %2
// CHECK-NEXT:       pdl_interp.is_not_null %4 : !pdl.value -> ^bb11, ^bb1
// CHECK-NEXT:     ^bb11:
// CHECK-NEXT:       %5 = pdl_interp.get_operand 1 of %2
// CHECK-NEXT:       pdl_interp.is_not_null %5 : !pdl.value -> ^bb12, ^bb1
// CHECK-NEXT:     ^bb12:
// CHECK-NEXT:       %6 = pdl_interp.get_result 0 of %2
// CHECK-NEXT:       pdl_interp.is_not_null %6 : !pdl.value -> ^bb13, ^bb1
// CHECK-NEXT:     ^bb13:
// CHECK-NEXT:       pdl_interp.are_equal %6, %1 : !pdl.value -> ^bb14, ^bb1
// CHECK-NEXT:     ^bb14:
// CHECK-NEXT:       %7 = pdl_interp.get_value_type of %6 : !pdl.type
// CHECK-NEXT:       %8 = pdl_interp.get_value_type of %0 : !pdl.type
// CHECK-NEXT:       pdl_interp.are_equal %7, %8 : !pdl.type -> ^bb15, ^bb1
// CHECK-NEXT:     ^bb15:
// CHECK-NEXT:       pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%5, %3, %7, %4, [[arg0]] : !pdl.value, !pdl.value, !pdl.type, !pdl.value, !pdl.operation) : benefit(1), generatedOps(["arith.subi", "arith.addi"]), loc([%2, [[arg0]]]), root("arith.subi") -> ^bb1
// CHECK-NEXT:     ^bb16:
// CHECK-NEXT:       pdl_interp.switch_operation_name of [[arg0]] to ["foo.op", "bar.op"](^bb1, ^bb17) -> ^bb2
// CHECK-NEXT:     ^bb17:
// CHECK-NEXT:       pdl_interp.check_type %7 is i32 -> ^bb15, ^bb1
// CHECK-NEXT:     ^bb18:
// CHECK-NEXT:       %9 = pdl_interp.get_attribute "test_attr" of [[arg0]]
// CHECK-NEXT:       pdl_interp.switch_attribute %9 to [42 : i32, true](^bb19, ^bb1) -> ^bb1
// CHECK-NEXT:     ^bb19:
// CHECK-NEXT:       %10 = pdl_interp.apply_constraint "myConstraint"(%9 : !pdl.attribute) : !pdl.operation {isNegated = true} -> ^bb15, ^bb1
// CHECK-NEXT:     ^bb20:
// CHECK-NEXT:       pdl_interp.apply_rewrite "myRewriter"([[arg0]] : !pdl.operation)
// CHECK-NEXT:       pdl_interp.finalize
// CHECK-NEXT:     ^bb21:
// CHECK-NEXT:       %11 = pdl_interp.apply_rewrite "myRewriter"([[arg0]] : !pdl.operation) : !pdl.operation
// CHECK-NEXT:       pdl_interp.finalize
// CHECK-NEXT:     ^bb22:
// CHECK-NEXT:       %12 = pdl_interp.apply_rewrite "myRewriter"([[arg0]], %0 : !pdl.operation, !pdl.value) : !pdl.attribute
// CHECK-NEXT:       pdl_interp.finalize
// CHECK-NEXT:     }
// CHECK-NEXT:     builtin.module @rewriters {
// CHECK-NEXT:       pdl_interp.func @pdl_generated_rewriter(%arg0 : !pdl.value, %arg1 : !pdl.value, %arg2 : !pdl.type, %arg3 : !pdl.value, %arg4 : !pdl.operation) {
// CHECK-NEXT:         %0 = pdl_interp.create_attribute 10 : i64
// CHECK-NEXT:         %1 = pdl_interp.create_type i64
// CHECK-NEXT:         %2 = pdl_interp.create_types [i32, i64]
// CHECK-NEXT:         %3 = pdl_interp.create_operation "arith.subi"(%arg0, %arg1 : !pdl.value, !pdl.value) {"attrA" = %0} -> (%arg2 : !pdl.type)
// CHECK-NEXT:         %4 = pdl_interp.create_operation "test.testop" {"attrA" = %0} -> (%arg2 : !pdl.type)
// CHECK-NEXT:         %5 = pdl_interp.get_result 0 of %3
// CHECK-NEXT:         %6 = pdl_interp.create_operation "arith.addi"(%arg3, %5 : !pdl.value, !pdl.value) -> (%arg2 : !pdl.type)
// CHECK-NEXT:         %7 = pdl_interp.get_result 0 of %6
// CHECK-NEXT:         %8 = pdl_interp.get_results of %6 : !pdl.range<value>
// CHECK-NEXT:         %9 = pdl_interp.get_results 0 of %6 : !pdl.range<value>
// CHECK-NEXT:         pdl_interp.replace %arg4 with (%8 : !pdl.range<value>)
// CHECK-NEXT:         pdl_interp.finalize
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
