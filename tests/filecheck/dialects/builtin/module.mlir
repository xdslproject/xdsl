// RUN: XDSL_ROUNDTRIP
// RUN: xdsl-opt %s --allow-unregistered-dialect | filecheck %s

builtin.module {
  // CHECK: builtin.module {
    builtin.module {}
    // CHECK: builtin.module {
    // CHECK-NEXT: }
    builtin.module attributes {a = "foo", b = "bar", unit} {}
    // CHECK-NEXT: builtin.module attributes {a = "foo", b = "bar", unit} {
    // CHECK-NEXT: }
    builtin.module @moduleName {}
    // CHECK-NEXT: builtin.module @moduleName {
    // CHECK-NEXT: }
    builtin.module @otherModule attributes {dialect.attr} {}
    // CHECK-NEXT: builtin.module @otherModule attributes  {dialect.attr} {
    // CHECK-NEXT: }
    module {}
    // CHECK-NEXT: builtin.module {
    // CHECK-NEXT: }
}
// CHECK: }
