// RUN: xdsl-opt %s | xdsl-opt | filecheck %s

builtin.module {
  // CHECK: builtin.module {
    builtin.module {}
    // CHECK: builtin.module {
    // CHECK-NEXT: }
    builtin.module attributes {a = "foo", b = "bar", unit} {}
    // CHECK-NEXT: builtin.module attributes {"a" = "foo", "b" = "bar", "unit"} {
    // CHECK-NEXT: }
}
// CHECK: }
