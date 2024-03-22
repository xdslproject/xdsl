// RUN: XDSL_ROUNDTRIP

%a = "test.op"() : () -> !x86.reg<rax>
// CHECK: %{{.*}} = "test.op"() : () -> !x86.reg<rax>
