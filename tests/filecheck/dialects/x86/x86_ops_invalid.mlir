// RUN: xdsl-opt --verify-diagnostics %s | filecheck %s

x86_func.func @funcyasm() {
    %3 = x86.get_register : () -> !x86.reg<rax>
    %4 = x86.get_register : () -> !x86.reg<rdx>
    %rflags = x86.ss.cmp %3, %4 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.rflags<rflags>
    x86.fallthrough ^fallthrough()
    // CHECK: Operation does not verify: Fallthrough op successor must immediately follow its parent.
    ^other:
    x86.label "other"
    ^fallthrough:
    x86.label "fallthrough"
    x86_func.ret
}
