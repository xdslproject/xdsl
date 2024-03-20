// RUN: XDSL_ROUNDTRIP
%a = "test.op"() : () -> !x86.reg<rax>
// CHECK: %{{.*}} = "test.op"() : () -> !x86.reg<rax>

%0 = x86.get_register : () -> !x86.reg<>
%1 = x86.get_register : () -> !x86.reg<>

%add = x86.add %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK: %{{.*}} = x86.add %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
