// RUN: xdsl-opt %s --parsing-diagnostics --split-input-file | filecheck %s

// A correct operation

builtin.module {
  %0 = "test.op"() : () -> !test.type<"foo">
  %1 = "test.op"(%0) : (!test.type<"foo">) -> !test.type<"bar">
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> !test.type<"foo">
// CHECK-NEXT:   %1 = "test.op"(%0) : (!test.type<"foo">) -> !test.type<"bar">
// CHECK-NEXT: }

// -----

// A correct operation with tuple values

builtin.module {
  %0 = "test.op"() : () -> !test.type<"foo">
  %1:2 = "test.op"(%0) : (!test.type<"foo">) -> (!test.type<"bar">, !test.type<"foo">)
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> !test.type<"foo">
// CHECK-NEXT:   %1, %2 = "test.op"(%0) : (!test.type<"foo">) -> (!test.type<"bar">, !test.type<"foo">)
// CHECK-NEXT: }

// -----

// An operation signature with not enough results

builtin.module {
  %0 = "test.op"() : () -> ()
}
// CHECK: Operation has 0 results, but were given 1 to bind.

// -----

// An operation signature with too many results

builtin.module {
  %0 = "test.op"() : () -> (!test.type<"foo">, !test.type<"bar">)
}

// CHECK: Operation has 2 results, but were given 1 to bind.

// -----

// An operation signature that has not enough operands

builtin.module {
  %0 = "test.op"() : () -> !test.type<"foo">
  "test.op"(%0) : () -> ()
}

// CHECK: expected 0 operand types but had 1

// -----

// An operation signature that has too many operands

builtin.module {
  "test.op"() : (i32) -> ()
}

// CHECK: expected 1 operand types but had 0

// -----

// An operation signature that doesn't have the correct operand type

builtin.module {
  %0 = "test.op"() : () -> !test.type<"foo">
  %1 = "test.op"(%0) : (!test.type<"bar">) -> !test.type<"bar">
}

// CHECK: mismatch between operand types and operation signature for operand #0. Expected !test.type<"bar"> but got !test.type<"foo">.

// -----

// An operation signature using tuples that doesn't have the correct operand type

builtin.module {
  %0:2 = "test.op"() : () -> (!test.type<"foo1">, !test.type<"foo2">)
  %1 = "test.op"(%0#1) : (!test.type<"bar">) -> !test.type<"bar">
}

// CHECK: mismatch between operand types and operation signature for operand #0. Expected !test.type<"bar"> but got !test.type<"foo2">.

// -----

// A forward declared operand that is used with an incorrect type

builtin.module {
  %1 = "test.op"(%0) : (!test.type<"bar">) -> !test.type<"bar">
  %0 = "test.op"() : () -> !test.type<"foo">
}

// CHECK: Result %0 is defined with type !test.type<"foo">, but used with type !test.type<"bar">

