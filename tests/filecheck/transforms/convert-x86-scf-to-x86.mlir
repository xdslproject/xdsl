// RUN: xdsl-opt -p convert-x86-scf-to-x86 --split-input-file --verify-diagnostics %s | filecheck %s


// CHECK-LABEL:    @copy10
//  CHECK-NEXT:      %zero = x86.di.mov 0 : () -> !x86.reg64<rcx>
//  CHECK-NEXT:      %step = x86.di.mov 4 : () -> !x86.reg64<rdx>
//  CHECK-NEXT:      %forty = x86.di.mov 40 : () -> !x86.reg64<r8>
//  CHECK-NEXT:      %0 = x86.ss.cmp %zero, %forty : (!x86.reg64<rcx>, !x86.reg64<r8>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jge %0 : !x86.rflags<rflags>, ^bb2(%zero : !x86.reg64<rcx>), ^bb1(%zero : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb1(%offset: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86.label "scf_body_0_for"
//  CHECK-NEXT:      "test.op"(%offset, %src, %dst) : (!x86.reg64<rcx>, !x86.reg64<rax>, !x86.reg64<rbx>) -> ()
//  CHECK-NEXT:      %offset_1 = x86.rs.add %offset, %step : (!x86.reg64<rcx>, !x86.reg64<rdx>) -> !x86.reg64<rcx>
//  CHECK-NEXT:      %1 = x86.ss.cmp %offset_1, %forty : (!x86.reg64<rcx>, !x86.reg64<r8>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jl %1 : !x86.rflags<rflags>, ^bb1(%offset_1 : !x86.reg64<rcx>), ^bb2(%offset_1 : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb2(%zero_end: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86.label "scf_body_end_0_for"
//  CHECK-NEXT:      x86_func.ret
//  CHECK-NEXT:    }
x86_func.func @copy10(%src: !x86.reg64<rax>, %dst: !x86.reg64<rbx>) {
    %zero = x86.di.mov 0 : () -> !x86.reg64<rcx>
    %step = x86.di.mov 4 : () -> !x86.reg64<rdx>
    %forty = x86.di.mov 40 : () -> !x86.reg64<r8>
    %zero_end = x86_scf.for %offset : !x86.reg64<rcx> = %zero to %forty step %step {
        "test.op"(%offset, %src, %dst) :  (!x86.reg64<rcx>, !x86.reg64<rax>, !x86.reg64<rbx>) -> ()
        yield
    }
    ret
}

// -----

// CHECK-LABEL:    x86_func.func @nested(%src: !x86.reg64<rax>, %dst: !x86.reg64<rbx>) {
//  CHECK-NEXT:      %zero_outer = x86.di.mov 0 : () -> !x86.reg64<rcx>
//  CHECK-NEXT:      %step_outer = x86.di.mov 4 : () -> !x86.reg64<rdx>
//  CHECK-NEXT:      %forty_outer = x86.di.mov 40 : () -> !x86.reg64<r8>
//  CHECK-NEXT:      %0 = x86.ss.cmp %zero_outer, %forty_outer : (!x86.reg64<rcx>, !x86.reg64<r8>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jge %0 : !x86.rflags<rflags>, ^bb4(%zero_outer : !x86.reg64<rcx>), ^bb1(%zero_outer : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb1(%offset_outer: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86.label "scf_body_1_for"
//  CHECK-NEXT:      %zero_inner = x86.di.mov 0 : () -> !x86.reg64<r10>
//  CHECK-NEXT:      %step_inner = x86.di.mov 2 : () -> !x86.reg64<r11>
//  CHECK-NEXT:      %forty_inner = x86.di.mov 40 : () -> !x86.reg64<r12>
//  CHECK-NEXT:      %1 = x86.ss.cmp %zero_inner, %forty_inner : (!x86.reg64<r10>, !x86.reg64<r12>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jge %1 : !x86.rflags<rflags>, ^bb3(%zero_inner : !x86.reg64<r10>), ^bb2(%zero_inner : !x86.reg64<r10>)
//  CHECK-NEXT:    ^bb2(%offset_inner: !x86.reg64<r10>):
//  CHECK-NEXT:      x86.label "scf_body_0_for"
//  CHECK-NEXT:      "test.op"(%src, %dst, %offset_outer, %offset_inner) : (!x86.reg64<rax>, !x86.reg64<rbx>, !x86.reg64<rcx>, !x86.reg64<r10>) -> ()
//  CHECK-NEXT:      %offset_inner_1 = x86.rs.add %offset_inner, %step_inner : (!x86.reg64<r10>, !x86.reg64<r11>) -> !x86.reg64<r10>
//  CHECK-NEXT:      %2 = x86.ss.cmp %offset_inner_1, %forty_inner : (!x86.reg64<r10>, !x86.reg64<r12>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jl %2 : !x86.rflags<rflags>, ^bb2(%offset_inner_1 : !x86.reg64<r10>), ^bb3(%offset_inner_1 : !x86.reg64<r10>)
//  CHECK-NEXT:    ^bb3(%zero_inner_end: !x86.reg64<r10>):
//  CHECK-NEXT:      x86.label "scf_body_end_0_for"
//  CHECK-NEXT:      %offset_outer_1 = x86.rs.add %offset_outer, %step_outer : (!x86.reg64<rcx>, !x86.reg64<rdx>) -> !x86.reg64<rcx>
//  CHECK-NEXT:      %3 = x86.ss.cmp %offset_outer_1, %forty_outer : (!x86.reg64<rcx>, !x86.reg64<r8>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jl %3 : !x86.rflags<rflags>, ^bb1(%offset_outer_1 : !x86.reg64<rcx>), ^bb4(%offset_outer_1 : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb4(%zero_outer_end: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86.label "scf_body_end_1_for"
//  CHECK-NEXT:      x86_func.ret
//  CHECK-NEXT:    }
//  CHECK-NEXT:  }



x86_func.func @nested(%src: !x86.reg64<rax>, %dst: !x86.reg64<rbx>) {
    %zero_outer = x86.di.mov 0 : () -> !x86.reg64<rcx>
    %step_outer = x86.di.mov 4 : () -> !x86.reg64<rdx>
    %forty_outer = x86.di.mov 40 : () -> !x86.reg64<r8>
    %zero_outer_end = x86_scf.for %offset_outer : !x86.reg64<rcx> = %zero_outer to %forty_outer step %step_outer {
        %zero_inner = x86.di.mov 0 : () -> !x86.reg64<r10>
        %step_inner = x86.di.mov 2 : () -> !x86.reg64<r11>
        %forty_inner = x86.di.mov 40 : () -> !x86.reg64<r12>
        %zero_inner_end = x86_scf.for %offset_inner : !x86.reg64<r10> = %zero_inner to %forty_inner step %step_inner {
            "test.op"(%src, %dst, %offset_outer, %offset_inner) : (!x86.reg64<rax>, !x86.reg64<rbx>, !x86.reg64<rcx>, !x86.reg64<r10>) -> ()
            x86_scf.yield
        }
        x86_scf.yield
    }
    ret
}

// -----
// Static upper bound is currently unsupported by this lowering.
x86_func.func @static_ub_for_fails() {
    %zero = x86.di.mov 0 : () -> !x86.reg64<rcx>
    %step = x86.di.mov 1 : () -> !x86.reg64<rdx>
    %zero_end = x86_scf.for %i : !x86.reg64<rcx> = %zero to 10 : si32 step %step {
        x86_scf.yield
    }
    ret
}
// CHECK: convert-x86-scf-to-x86 expects x86_scf.for upper bound to be an SSAValue

// -----
// Static upper bound remains unsupported even with iter_args.
x86_func.func @static_ub_for_with_iter_args_fails(%init: !x86.reg64<rax>) {
    %zero = x86.di.mov 0 : () -> !x86.reg64<rcx>
    %step = x86.di.mov 1 : () -> !x86.reg64<rdx>
    %zero_end, %res = x86_scf.for %i : !x86.reg64<rcx> = %zero to 20 : si32 step %step iter_args(%acc = %init) -> (!x86.reg64<rax>) {
        x86_scf.yield %acc : !x86.reg64<rax>
    }
    "test.op"(%res) : (!x86.reg64<rax>) -> ()
    ret
}
// CHECK: convert-x86-scf-to-x86 expects x86_scf.for upper bound to be an SSAValue

// -----
// Static step is currently unsupported by this lowering.
x86_func.func @static_step_for_fails() {
    %zero = x86.di.mov 0 : () -> !x86.reg64<rcx>
    %forty = x86.di.mov 40 : () -> !x86.reg64<r8>
    %zero_end = x86_scf.for %i : !x86.reg64<rcx> = %zero to %forty step 1 : si32 {
        x86_scf.yield
    }
    ret
}
// CHECK: convert-x86-scf-to-x86 expects x86_scf.for step to be an SSAValue

// -----
// Static step remains unsupported even with iter_args.
x86_func.func @static_step_for_with_iter_args_fails(%init: !x86.reg64<rax>) {
    %zero = x86.di.mov 0 : () -> !x86.reg64<rcx>
    %forty = x86.di.mov 40 : () -> !x86.reg64<r8>
    %zero_end, %res = x86_scf.for %i : !x86.reg64<rcx> = %zero to %forty step 1 : si32 iter_args(%acc = %init) -> (!x86.reg64<rax>) {
        x86_scf.yield %acc : !x86.reg64<rax>
    }
    "test.op"(%res) : (!x86.reg64<rax>) -> ()
    ret
}
// CHECK: convert-x86-scf-to-x86 expects x86_scf.for step to be an SSAValue
