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
