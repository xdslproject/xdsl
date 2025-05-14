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

pdl_interp.func @matcher(%arg2 : !pdl.operation) {
  %0 = pdl_interp.get_operand 1 of %arg2
  %1 = pdl_interp.get_defining_op of %0 : !pdl.value
  pdl_interp.is_not_null %1 : !pdl.operation -> ^0, ^1
^1:
  pdl_interp.finalize
^0:
  pdl_interp.check_operation_name of %arg2 is "arith.addi" -> ^2, ^1
^2:
  pdl_interp.check_operand_count of %arg2 is 2 -> ^3, ^1
^3:
  pdl_interp.check_result_count of %arg2 is 1 -> ^4, ^1
^4:
  %2 = pdl_interp.get_operand 0 of %arg2
  pdl_interp.is_not_null %2 : !pdl.value -> ^5, ^1
^5:
  pdl_interp.is_not_null %0 : !pdl.value -> ^6, ^1
^6:
  %3 = pdl_interp.get_result 0 of %arg2
  pdl_interp.is_not_null %3 : !pdl.value -> ^7, ^1
^7:
  pdl_interp.check_operation_name of %1 is "arith.constant" -> ^8, ^1
^8:
  pdl_interp.check_operand_count of %1 is 0 -> ^9, ^1
^9:
  pdl_interp.check_result_count of %1 is 1 -> ^10, ^1
^10:
  %4 = pdl_interp.get_attribute "value" of %1
  pdl_interp.is_not_null %4 : !pdl.attribute -> ^11, ^1
^11:
  pdl_interp.check_attribute %4 is 0 : i32 -> ^12, ^1
^12:
  %5 = pdl_interp.get_result 0 of %1
  pdl_interp.is_not_null %5 : !pdl.value -> ^13, ^1
^13:
  pdl_interp.are_equal %5, %0 : !pdl.value -> ^14, ^1
^14:
  %6 = pdl_interp.get_value_type of %5 : !pdl.type
  %7 = pdl_interp.get_value_type of %3 : !pdl.type
  pdl_interp.are_equal %6, %7 : !pdl.type -> ^15, ^1
^15:
  pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%2, %arg2 : !pdl.value, !pdl.operation) : benefit(2), loc([%1, %arg2]), root("arith.addi") -> ^1
}
builtin.module @rewriters {
  pdl_interp.func @pdl_generated_rewriter(%arg0 : !pdl.value, %arg1 : !pdl.operation) {
    pdl_interp.replace %arg1 with (%arg0 : !pdl.value)
    pdl_interp.finalize
  }
}
