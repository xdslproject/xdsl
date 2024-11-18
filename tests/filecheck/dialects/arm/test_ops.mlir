// RUN: XDSL_ROUNDTRIP


// CHECK: %x1 = "test.op"() : () -> (!arm.reg<x1>)
%x1 = "arm.get_register"() : () -> !arm.reg<x1>
