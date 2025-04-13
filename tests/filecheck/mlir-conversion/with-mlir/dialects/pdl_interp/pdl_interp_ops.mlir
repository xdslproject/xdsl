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
}
module @rewriters {
  pdl_interp.func @pdl_generated_rewriter(%arg0: !pdl.value, %arg1: !pdl.value, %arg2: !pdl.type, %arg3: !pdl.value, %arg4: !pdl.operation) {
    %attr = pdl_interp.create_attribute 10 : i64 
    %0 = pdl_interp.create_operation "arith.subi"(%arg0, %arg1 : !pdl.value, !pdl.value) {"attrA" = %attr}  -> (%arg2 : !pdl.type)
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
// CHECK-NEXT:       pdl_interp.is_not_null %0 : !pdl.value -> ^0, ^1
// CHECK-NEXT:     ^1:
// CHECK-NEXT:       pdl_interp.finalize
// CHECK-NEXT:     ^0:
// CHECK-NEXT:       %1 = pdl_interp.get_operand 0 of [[arg0]]
// CHECK-NEXT:       %2 = pdl_interp.get_defining_op of %1 : !pdl.value
// CHECK-NEXT:       pdl_interp.is_not_null %2 : !pdl.operation -> ^2, ^1
// CHECK-NEXT:     ^2:
// CHECK-NEXT:       pdl_interp.check_operation_name of [[arg0]] is "arith.subi" -> ^3, ^1
// CHECK-NEXT:     ^3:
// CHECK-NEXT:       pdl_interp.check_operand_count of [[arg0]] is 2 -> ^4, ^1
// CHECK-NEXT:     ^4:
// CHECK-NEXT:       pdl_interp.check_result_count of [[arg0]] is 1 -> ^5, ^1
// CHECK-NEXT:     ^5:
// CHECK-NEXT:       pdl_interp.is_not_null %1 : !pdl.value -> ^6, ^1
// CHECK-NEXT:     ^6:
// CHECK-NEXT:       %3 = pdl_interp.get_operand 1 of [[arg0]]
// CHECK-NEXT:       pdl_interp.is_not_null %3 : !pdl.value -> ^7, ^1
// CHECK-NEXT:     ^7:
// CHECK-NEXT:       pdl_interp.check_operation_name of %2 is "arith.addi" -> ^8, ^1
// CHECK-NEXT:     ^8:
// CHECK-NEXT:       pdl_interp.check_operand_count of %2 is 2 -> ^9, ^1
// CHECK-NEXT:     ^9:
// CHECK-NEXT:       pdl_interp.check_result_count of %2 is 1 -> ^10, ^1
// CHECK-NEXT:     ^10:
// CHECK-NEXT:       %4 = pdl_interp.get_operand 0 of %2
// CHECK-NEXT:       pdl_interp.is_not_null %4 : !pdl.value -> ^11, ^1
// CHECK-NEXT:     ^11:
// CHECK-NEXT:       %5 = pdl_interp.get_operand 1 of %2
// CHECK-NEXT:       pdl_interp.is_not_null %5 : !pdl.value -> ^12, ^1
// CHECK-NEXT:     ^12:
// CHECK-NEXT:       %6 = pdl_interp.get_result 0 of %2
// CHECK-NEXT:       pdl_interp.is_not_null %6 : !pdl.value -> ^13, ^1
// CHECK-NEXT:     ^13:
// CHECK-NEXT:       pdl_interp.are_equal %6, %1 : !pdl.value -> ^14, ^1
// CHECK-NEXT:     ^14:
// CHECK-NEXT:       %7 = pdl_interp.get_value_type of %6 : !pdl.type
// CHECK-NEXT:       %8 = pdl_interp.get_value_type of %0 : !pdl.type
// CHECK-NEXT:       pdl_interp.are_equal %7, %8 : !pdl.type -> ^15, ^1
// CHECK-NEXT:     ^15:
// CHECK-NEXT:       pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%5, %3, %7, %4, [[arg0]] : !pdl.value, !pdl.value, !pdl.type, !pdl.value, !pdl.operation) : benefit(1), generatedOps(["arith.subi", "arith.addi"]), loc([%2, [[arg0]]]), root("arith.subi") -> ^1
// CHECK-NEXT:     }
// CHECK-NEXT:     builtin.module @rewriters {
// CHECK-NEXT:       pdl_interp.func @pdl_generated_rewriter(%arg0 : !pdl.value, %arg1 : !pdl.value, %arg2 : !pdl.type, %arg3 : !pdl.value, %arg4 : !pdl.operation) {
// CHECK-NEXT:         %0 = pdl_interp.create_attribute 10 : i64
// CHECK-NEXT:         %1 = pdl_interp.create_operation "arith.subi"(%arg0, %arg1 : !pdl.value, !pdl.value) {"attrA" = %0} -> (%arg2 : !pdl.type)
// CHECK-NEXT:         %2 = pdl_interp.get_result 0 of %1
// CHECK-NEXT:         %3 = pdl_interp.create_operation "arith.addi"(%arg3, %2 : !pdl.value, !pdl.value) -> (%arg2 : !pdl.type)
// CHECK-NEXT:         %4 = pdl_interp.get_result 0 of %3
// CHECK-NEXT:         %5 = pdl_interp.get_results of %3 : !pdl.range<value>
// CHECK-NEXT:         %6 = pdl_interp.get_results 0 of %3 : !pdl.range<value>
// CHECK-NEXT:         pdl_interp.replace %arg4 with (%5 : !pdl.range<value>)
// CHECK-NEXT:         pdl_interp.finalize
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
