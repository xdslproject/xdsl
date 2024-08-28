// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK:       builtin.module {
// CHECK-GENERIC:       "builtin.module"() ({

stim.circuit {}
// CHECK-NEXT:    stim.circuit {
// CHECK-GENERIC-NEXT:    "stim.circuit"() ({
// CHECK-NEXT: }
// CHECK-GENERIC-NEXT:   }) : () -> ()

stim.circuit attributes {"hello" = "world"} {}
// CHECK-NEXT:    stim.circuit attributes {"hello" = "world"} {
// CHECK-GENERIC-NEXT:    "stim.circuit"() ({
// CHECK-NEXT: }
// CHECK-GENERIC-NEXT:   }) {"hello" = "world"} : () -> ()

// CHECK-NEXT:  }
// CHECK-GENERIC-NEXT:  }) : () -> ()


