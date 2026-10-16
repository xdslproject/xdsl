// RUN: XDSL_ROUNDTRIP

builtin.module @rewriters {
  builtin.module @rewrite {}
}

// CHECK: match.matcher "private" @named root(%root: !pdl.operation) {
match.matcher "private" @named root(%root : !pdl.operation) {
  // CHECK-NEXT: match.has_name %root, "arith.addi"
  match.has_name %root, "arith.addi"
  // CHECK-NEXT: match.success @rewriters::@rewrite benefit(3) (%root : !pdl.operation)
  match.success @rewriters::@rewrite benefit(3) (%root : !pdl.operation)
  // Success is not a terminator.
  // CHECK-NEXT: match.has_name %root, "arith.addi"
  match.has_name %root, "arith.addi"
// CHECK-NEXT: } attributes {tag = "named"}
} attributes {tag = "named"}

// CHECK: match.matcher root(%empty: !pdl.operation) {
match.matcher root(%empty : !pdl.operation) {
  // CHECK-NEXT: match.success @rewriters::@rewrite benefit(0)
  match.success @rewriters::@rewrite benefit(0)
}

// CHECK: match.matcher @captures
match.matcher @captures root(%op : !pdl.operation) {
  %attr, %type, %value, %range = "test.op"() : () -> (!pdl.attribute, !pdl.type, !pdl.value, !pdl.range<value>)
  // CHECK: match.success @rewriters::@rewrite benefit(1) (%op, %attr, %type, %value, %range : !pdl.operation, !pdl.attribute, !pdl.type, !pdl.value, !pdl.range<value>)
  match.success @rewriters::@rewrite benefit(1) (%op, %attr, %type, %value, %range : !pdl.operation, !pdl.attribute, !pdl.type, !pdl.value, !pdl.range<value>)
}

// CHECK: match.matcher @navigation
match.matcher @navigation root(%nav : !pdl.operation) {
  // CHECK: %operand = match.get_operand 2 of %nav : !match.optional<!pdl.value>
  // Also accept the elided operation names printed by MLIR inside a matcher.
  %operand = get_operand 2 of %nav : !match.optional<!pdl.value>
  // CHECK: %result = match.get_result 1 of %nav : !match.optional<!pdl.value>
  %result = match.get_result 1 of %nav : !match.optional<!pdl.value>
  // CHECK: %attribute = match.get_attribute "value" of %nav : !match.optional<!pdl.attribute>
  %attribute = match.get_attribute "value" of %nav : !match.optional<!pdl.attribute>
  // CHECK: %unwrapped = match.is_not_null %operand : !pdl.value
  %unwrapped = match.is_not_null %operand : !pdl.value
  // CHECK: %def = match.get_defining_op of %unwrapped : !pdl.value -> !match.optional<!pdl.operation>
  %def = match.get_defining_op of %unwrapped : !pdl.value -> !match.optional<!pdl.operation>
  %maybe_range = "test.op"() : () -> !match.optional<!pdl.range<value>>
  // CHECK: %values = match.is_not_null %maybe_range : !pdl.range<value>
  %values = match.is_not_null %maybe_range : !pdl.range<value>
  // CHECK: %range_def = match.get_defining_op of %values : !pdl.range<value> -> !match.optional<!pdl.operation>
  %range_def = match.get_defining_op of %values : !pdl.range<value> -> !match.optional<!pdl.operation>
  match.success @rewriters::@rewrite benefit(1) (%unwrapped, %values : !pdl.value, !pdl.range<value>)
}

// CHECK: func.func private @optionals(!match.optional<!pdl.attribute>, !match.optional<!pdl.operation>, !match.optional<!pdl.type>, !match.optional<!pdl.value>, !match.optional<!pdl.range<attribute>>, !match.optional<!pdl.range<operation>>, !match.optional<!pdl.range<type>>, !match.optional<!pdl.range<value>>)
func.func private @optionals(
  !match.optional<!pdl.attribute>, !match.optional<!pdl.operation>,
  !match.optional<!pdl.type>, !match.optional<!pdl.value>,
  !match.optional<!pdl.range<attribute>>, !match.optional<!pdl.range<operation>>,
  !match.optional<!pdl.range<type>>, !match.optional<!pdl.range<value>>)
