// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP


// CHECK: %x1 = arm.get_register : !arm.reg<x1>
%x1 = arm.get_register : !arm.reg<x1>


// CHECK-GENERIC: %x1 = "arm.get_register"() : () -> !arm.reg<x1>
