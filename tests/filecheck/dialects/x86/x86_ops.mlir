// RUN: XDSL_ROUNDTRIP

"builtin.module"() ({
    %a = "test.op"() : () -> !x86.reg<rax>
    // CHECK: %{{.*}} = "test.op"() : () -> !x86.reg<rax>
}) : () -> ()