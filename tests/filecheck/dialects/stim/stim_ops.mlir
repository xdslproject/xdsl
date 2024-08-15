// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK:       builtin.module {
// CHECK-GENERIC:       "builtin.module"() ({


stim.circuit attributes {} {}
// CHECK-NEXT:    stim.circuit attributes {} {}
// CHECK-GENERIC-NEXT:    "stim.circuit"() {} {} : () -> ()


// CHECK-NEXT:  }
// CHECK-GENERIC-NEXT:  }) : () -> ()

