// RUN: XDSL_ROUNDTRIP

builtin.module {
  // CHECK: builtin.module {
    builtin.module {}
    // CHECK: builtin.module {
    // CHECK-NEXT: }
    builtin.module attributes {a = "foo", b = "bar", unit} {}
    // CHECK-NEXT: builtin.module attributes {"a" = "foo", "b" = "bar", "unit"} {
    // CHECK-NEXT: }
    builtin.module @moduleName {}
    // CHECK-NEXT: builtin.module @moduleName {
    // CHECK-NEXT: }
    builtin.module @otherModule attributes {dialect.attr} {}
    // CHECK-NEXT: builtin.module @otherModule attributes  {"dialect.attr"} {
    // CHECK-NEXT: }
}
// CHECK: }
