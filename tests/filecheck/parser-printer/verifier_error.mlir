// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

builtin.module {
    // Operation has an unexpected result
    %x = "builtin.module"() : () -> (i32)
}

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   %x = "builtin.module"() : () -> i32
// CHECK-NEXT:   ^^^^^^^^^^^^^^^^^^^^^------------------------------------
// CHECK-NEXT:   | Operation does not verify: Expected 0 results, but got 1
// CHECK-NEXT:   ---------------------------------------------------------
// CHECK-NEXT: }) : () -> ()
