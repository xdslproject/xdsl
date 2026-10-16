// RUN: XDSL_ROUNDTRIP

// A verified Match description of arith.addi(x, 0 : i32) -> x.
// This fixture checks syntax and verification, not execution.

builtin.module @rewriters {
  // CHECK: pdl_interp.func @add_zero(%target: !pdl.operation, %replacement: !pdl.value) {
  pdl_interp.func @add_zero(%target: !pdl.operation, %replacement: !pdl.value) {
    // CHECK-NEXT: pdl_interp.replace %target with (%replacement : !pdl.value)
    pdl_interp.replace %target with (%replacement : !pdl.value)
    // CHECK-NEXT: pdl_interp.finalize
    pdl_interp.finalize
  }
}

// CHECK: match.matcher @add_zero root(%root: !pdl.operation) {
match.matcher @add_zero root(%root : !pdl.operation) {
  // CHECK-NEXT: match.has_name %root, "arith.addi"
  match.has_name %root, "arith.addi"
  // CHECK-NEXT: match.check_operand_count %root is 2
  match.check_operand_count %root is 2
  // CHECK-NEXT: match.check_result_count %root is 1
  match.check_result_count %root is 1
  // CHECK-NEXT: %maybe_lhs = match.get_operand 0 of %root : !match.optional<!pdl.value>
  %maybe_lhs = match.get_operand 0 of %root : !match.optional<!pdl.value>
  // CHECK-NEXT: %lhs = match.is_not_null %maybe_lhs : !pdl.value
  %lhs = match.is_not_null %maybe_lhs : !pdl.value
  // CHECK-NEXT: %maybe_rhs = match.get_operand 1 of %root : !match.optional<!pdl.value>
  %maybe_rhs = match.get_operand 1 of %root : !match.optional<!pdl.value>
  // CHECK-NEXT: %rhs = match.is_not_null %maybe_rhs : !pdl.value
  %rhs = match.is_not_null %maybe_rhs : !pdl.value
  // CHECK-NEXT: %maybe_constant = match.get_defining_op of %rhs : !pdl.value -> !match.optional<!pdl.operation>
  %maybe_constant = match.get_defining_op of %rhs : !pdl.value -> !match.optional<!pdl.operation>
  // CHECK-NEXT: %constant = match.is_not_null %maybe_constant : !pdl.operation
  %constant = match.is_not_null %maybe_constant : !pdl.operation
  // CHECK-NEXT: match.has_name %constant, "arith.constant"
  match.has_name %constant, "arith.constant"
  // CHECK-NEXT: %maybe_value = match.get_attribute "value" of %constant : !match.optional<!pdl.attribute>
  %maybe_value = match.get_attribute "value" of %constant : !match.optional<!pdl.attribute>
  // CHECK-NEXT: %value = match.is_not_null %maybe_value : !pdl.attribute
  %value = match.is_not_null %maybe_value : !pdl.attribute
  // CHECK-NEXT: match.has_attr_value %value is 0 : i32
  match.has_attr_value %value is 0 : i32
  // CHECK-NEXT: match.success @rewriters::@add_zero benefit(1) (%root, %lhs : !pdl.operation, !pdl.value)
  match.success @rewriters::@add_zero benefit(1) (%root, %lhs : !pdl.operation, !pdl.value)
}
