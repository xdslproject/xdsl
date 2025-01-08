// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK:       builtin.module {
// CHECK-GENERIC:       "builtin.module"() ({


wasm.module
// CHECK-NEXT:    wasm.module
// CHECK-GENERIC-NEXT:    "wasm.module"() : () -> ()

wasm.module attributes {"hello" = "world"}
// CHECK-NEXT:    wasm.module attributes {hello = "world"}
// CHECK-GENERIC-NEXT:    "wasm.module"() {hello = "world"} : () -> ()


// CHECK-NEXT:  }
// CHECK-GENERIC-NEXT:  }) : () -> ()
