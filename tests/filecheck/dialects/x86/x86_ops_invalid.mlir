// RUN: xdsl-opt --verify-diagnostics --split-input-file %s | filecheck %s

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

// -----

// Mismatched input and output counts
%0, %1, %2 = "test.op"() : () -> (!x86.reg<rax>, !x86.reg<rbx>, !x86.reg<rcx>)
x86.parallel_mov %0, %1, %2 : (!x86.reg<rax>, !x86.reg<rbx>, !x86.reg<rcx>) -> (!x86.reg<rdx>, !x86.reg<rsi>)

// CHECK: %0, %1, %2 = "test.op"() : () -> (!x86.reg<rax>, !x86.reg<rbx>, !x86.reg<rcx>)
// CHECK: Input count must match output count. Num inputs: 3, Num outputs: 2

// -----

// Moving from int register to vector register
%0, %1 = "test.op"() : () -> (!x86.reg<r8>, !x86.reg<r9>)
x86.parallel_mov %0, %1 : (!x86.reg<r8>, !x86.reg<r9>) -> (!x86.ssereg, !x86.ssereg)

// CHECK: %0, %1 = "test.op"() : () -> (!x86.reg<r8>, !x86.reg<r9>)
// CHECK: Input type must match output type.

// -----

// Duplicated output registers
%0, %1 = "test.op"() : () -> (!x86.reg<r10>, !x86.reg<r11>)
x86.parallel_mov %0, %1 : (!x86.reg<r10>, !x86.reg<r11>) -> (!x86.reg<r12>, !x86.reg<r12>)

// CHECK: %0, %1 = "test.op"() : () -> (!x86.reg<r10>, !x86.reg<r11>)
// CHECK: Outputs must be unallocated or distinct.
