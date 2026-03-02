// RUN: xdsl-opt -p convert-x86-scf-to-x86 --split-input-file %s | filecheck %s


// CHECK-LABEL:    @copy10
//  CHECK-NEXT:      %zero = x86.di.mov 0 : () -> !x86.reg<rcx>
//  CHECK-NEXT:      %step = x86.di.mov 4 : () -> !x86.reg<rdx>
//  CHECK-NEXT:      %forty = x86.di.mov 40 : () -> !x86.reg<r8>
//  CHECK-NEXT:      %zero_1 = x86.ds.mov %zero : (!x86.reg<rcx>) -> !x86.reg<r9>
//  CHECK-NEXT:      %0 = x86.ss.cmp %zero_1, %forty : (!x86.reg<r9>, !x86.reg<r8>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jge %0 : !x86.rflags<rflags>, ^bb0(%zero_1 : !x86.reg<r9>), ^bb1(%zero_1 : !x86.reg<r9>)
//  CHECK-NEXT:    ^bb1(%offset : !x86.reg<r9>):
//  CHECK-NEXT:      x86.label "scf_body_0_for"
//  CHECK-NEXT:      "test.op"(%offset, %src, %dst) : (!x86.reg<r9>, !x86.reg<rax>, !x86.reg<rbx>) -> ()
//  CHECK-NEXT:      %offset_1 = x86.ds.mov %offset : (!x86.reg<r9>) -> !x86.reg<r9>
//  CHECK-NEXT:      %offset_2 = x86.r.inc %offset_1 : (!x86.reg<r9>) -> !x86.reg<r9>
//  CHECK-NEXT:      %1 = x86.ss.cmp %offset_2, %forty : (!x86.reg<r9>, !x86.reg<r8>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jl %1 : !x86.rflags<rflags>, ^bb1(%offset_2 : !x86.reg<r9>), ^bb0(%offset_2 : !x86.reg<r9>)
//  CHECK-NEXT:    ^bb0(%offset_3 : !x86.reg<r9>):
//  CHECK-NEXT:      x86.label "scf_body_end_0_for"
//  CHECK-NEXT:      x86_func.ret
//  CHECK-NEXT:    }
x86_func.func @copy10(%src : !x86.reg<rax>, %dst : !x86.reg<rbx>) {
    %zero = x86.di.mov 0 : () -> (!x86.reg<rcx>)
    %step = x86.di.mov 4 : () -> (!x86.reg<rdx>)
    %forty = x86.di.mov 40 : () -> (!x86.reg<r8>)
    x86_scf.for %offset : !x86.reg<r9> = %zero to %forty step %step {
        "test.op"(%offset, %src, %dst) :  (!x86.reg<r9>, !x86.reg<rax>, !x86.reg<rbx>) -> ()
        yield
    }
    ret
}

// -----

// CHECK-LABEL:    x86_func.func @nested(%src : !x86.reg<rax>, %dst : !x86.reg<rbx>) {
//  CHECK-NEXT:      %zero_outer = x86.di.mov 0 : () -> !x86.reg<rcx>
//  CHECK-NEXT:      %step_outer = x86.di.mov 4 : () -> !x86.reg<rdx>
//  CHECK-NEXT:      %forty_outer = x86.di.mov 40 : () -> !x86.reg<r8>
//  CHECK-NEXT:      %zero_outer_1 = x86.ds.mov %zero_outer : (!x86.reg<rcx>) -> !x86.reg<r9>
//  CHECK-NEXT:      %0 = x86.ss.cmp %zero_outer_1, %forty_outer : (!x86.reg<r9>, !x86.reg<r8>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jge %0 : !x86.rflags<rflags>, ^bb0(%zero_outer_1 : !x86.reg<r9>), ^bb1(%zero_outer_1 : !x86.reg<r9>)
//  CHECK-NEXT:    ^bb1(%offset_outer : !x86.reg<r9>):
//  CHECK-NEXT:      x86.label "scf_body_1_for"
//  CHECK-NEXT:      %zero_inner = x86.di.mov 0 : () -> !x86.reg<r10>
//  CHECK-NEXT:      %step_inner = x86.di.mov 2 : () -> !x86.reg<r11>
//  CHECK-NEXT:      %forty_inner = x86.di.mov 40 : () -> !x86.reg<r12>
//  CHECK-NEXT:      %zero_inner_1 = x86.ds.mov %zero_inner : (!x86.reg<r10>) -> !x86.reg<r13>
//  CHECK-NEXT:      %1 = x86.ss.cmp %zero_inner_1, %forty_inner : (!x86.reg<r13>, !x86.reg<r12>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jge %1 : !x86.rflags<rflags>, ^bb2(%zero_inner_1 : !x86.reg<r13>), ^bb3(%zero_inner_1 : !x86.reg<r13>)
//  CHECK-NEXT:    ^bb3(%offset_inner : !x86.reg<r13>):
//  CHECK-NEXT:      x86.label "scf_body_0_for"
//  CHECK-NEXT:      "test.op"(%src, %dst, %offset_outer, %offset_inner) : (!x86.reg<rax>, !x86.reg<rbx>, !x86.reg<r9>, !x86.reg<r13>) -> ()
//  CHECK-NEXT:      %offset_inner_1 = x86.ds.mov %offset_inner : (!x86.reg<r13>) -> !x86.reg<r13>
//  CHECK-NEXT:      %offset_inner_2 = x86.r.inc %offset_inner_1 : (!x86.reg<r13>) -> !x86.reg<r13>
//  CHECK-NEXT:      %2 = x86.ss.cmp %offset_inner_2, %forty_inner : (!x86.reg<r13>, !x86.reg<r12>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jl %2 : !x86.rflags<rflags>, ^bb3(%offset_inner_2 : !x86.reg<r13>), ^bb2(%offset_inner_2 : !x86.reg<r13>)
//  CHECK-NEXT:    ^bb2(%offset_inner_3 : !x86.reg<r13>):
//  CHECK-NEXT:      x86.label "scf_body_end_0_for"
//  CHECK-NEXT:      %offset_outer_1 = x86.ds.mov %offset_outer : (!x86.reg<r9>) -> !x86.reg<r9>
//  CHECK-NEXT:      %offset_outer_2 = x86.r.inc %offset_outer_1 : (!x86.reg<r9>) -> !x86.reg<r9>
//  CHECK-NEXT:      %3 = x86.ss.cmp %offset_outer_2, %forty_outer : (!x86.reg<r9>, !x86.reg<r8>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jl %3 : !x86.rflags<rflags>, ^bb1(%offset_outer_2 : !x86.reg<r9>), ^bb0(%offset_outer_2 : !x86.reg<r9>)
//  CHECK-NEXT:    ^bb0(%offset_outer_3 : !x86.reg<r9>):
//  CHECK-NEXT:      x86.label "scf_body_end_1_for"
//  CHECK-NEXT:      x86_func.ret
//  CHECK-NEXT:    }
//  CHECK-NEXT:  }



x86_func.func @nested(%src : !x86.reg<rax>, %dst : !x86.reg<rbx>) {
    %zero_outer = x86.di.mov 0 : () -> (!x86.reg<rcx>)
    %step_outer = x86.di.mov 4 : () -> (!x86.reg<rdx>)
    %forty_outer = x86.di.mov 40 : () -> (!x86.reg<r8>)
    x86_scf.for %offset_outer : !x86.reg<r9> = %zero_outer to %forty_outer step %step_outer {
        %zero_inner = x86.di.mov 0 : () -> (!x86.reg<r10>)
        %step_inner = x86.di.mov 2 : () -> (!x86.reg<r11>)
        %forty_inner = x86.di.mov 40 : () -> (!x86.reg<r12>)
        x86_scf.for %offset_inner : !x86.reg<r13> = %zero_inner to %forty_inner step %step_inner {
            "test.op"(%src, %dst, %offset_outer, %offset_inner) : (!x86.reg<rax>, !x86.reg<rbx>, !x86.reg<r9>, !x86.reg<r13>) -> ()
            x86_scf.yield
        }
        x86_scf.yield
    }
    ret
}
