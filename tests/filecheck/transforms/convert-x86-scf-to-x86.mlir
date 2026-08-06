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


// CHECK-LABEL:    @iv_not_used
//  CHECK-NEXT:      %zero = x86.di.mov 0 : () -> !x86.reg64<rcx>
//  CHECK-NEXT:      %step = x86.di.mov 4 : () -> !x86.reg64<rdx>
//  CHECK-NEXT:      %forty = x86.di.mov 40 : () -> !x86.reg64<r8>
//  CHECK-NEXT:      %0 = x86.ss.cmp %zero, %forty : (!x86.reg64<rcx>, !x86.reg64<r8>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jge %0 : !x86.rflags<rflags>, ^bb2(%zero : !x86.reg64<rcx>), ^bb1(%zero : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb1(%offset: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86.label "scf_body_0_for"
//  CHECK-NEXT:      %offset_1 = x86.rs.add %offset, %step : (!x86.reg64<rcx>, !x86.reg64<rdx>) -> !x86.reg64<rcx>
//  CHECK-NEXT:      "test.op"(%src, %dst) : (!x86.reg64<rax>, !x86.reg64<rbx>) -> ()
//  CHECK-NEXT:      %1 = x86.ss.cmp %offset_1, %forty : (!x86.reg64<rcx>, !x86.reg64<r8>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jl %1 : !x86.rflags<rflags>, ^bb1(%offset_1 : !x86.reg64<rcx>), ^bb2(%offset_1 : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb2(%offset_end: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86.label "scf_body_end_0_for"
//  CHECK-NEXT:      x86_func.ret
//  CHECK-NEXT:    }
x86_func.func @iv_not_used(%src: !x86.reg64<rax>, %dst: !x86.reg64<rbx>) {
    %zero = x86.di.mov 0 : () -> !x86.reg64<rcx>
    %step = x86.di.mov 4 : () -> !x86.reg64<rdx>
    %forty = x86.di.mov 40 : () -> !x86.reg64<r8>
    %offset_end = x86_scf.for %offset : !x86.reg64<rcx> = %zero to %forty step %step {
        "test.op"(%src, %dst) :  (!x86.reg64<rax>, !x86.reg64<rbx>) -> ()
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
// CHECK-LABEL:    x86_func.func @entry_fallthrough_static_ub_dyn_step() {
//  CHECK-NEXT:      %zero = x86.di.mov 0 : () -> !x86.reg64<rcx>
//  CHECK-NEXT:      %step = x86.di.mov 1 : () -> !x86.reg64<rdx>
//  CHECK-NEXT:      x86.fallthrough ^bb1(%zero : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb1(%i: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86.label "scf_body_0_for"
//  CHECK-NEXT:      %i_1 = x86.rs.add %i, %step : (!x86.reg64<rcx>, !x86.reg64<rdx>) -> !x86.reg64<rcx>
//  CHECK-NEXT:      %0 = x86.si.cmp %i_1, 10 : (!x86.reg64<rcx>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jl %0 : !x86.rflags<rflags>, ^bb1(%i_1 : !x86.reg64<rcx>), ^bb2(%i_1 : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb2(%zero_end: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86_func.ret
//  CHECK-NEXT:    }
x86_func.func @entry_fallthrough_static_ub_dyn_step() {
    %zero = x86.di.mov 0 : () -> !x86.reg64<rcx>
    %step = x86.di.mov 1 : () -> !x86.reg64<rdx>
    %zero_end = x86_scf.for %i : !x86.reg64<rcx> = %zero to 10 : si32 step %step {
        x86_scf.yield
    }
    ret
}

// -----
// CHECK-LABEL:    x86_func.func @entry_fallthrough_static_bounds() {
//  CHECK-NEXT:      %zero = x86.di.mov 0 : () -> !x86.reg64<rcx>
//  CHECK-NEXT:      x86.fallthrough ^bb1(%zero : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb1(%i: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86.label "scf_body_0_for"
//  CHECK-NEXT:      %i_1 = x86.ri.add %i, 1 : (!x86.reg64<rcx>) -> !x86.reg64<rcx>
//  CHECK-NEXT:      %0 = x86.si.cmp %i_1, 12 : (!x86.reg64<rcx>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jl %0 : !x86.rflags<rflags>, ^bb1(%i_1 : !x86.reg64<rcx>), ^bb2(%i_1 : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb2(%zero_end: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86_func.ret
//  CHECK-NEXT:    }
x86_func.func @entry_fallthrough_static_bounds() {
    %zero = x86.di.mov 0 : () -> !x86.reg64<rcx>
    %zero_end = x86_scf.for %i : !x86.reg64<rcx> = %zero to 12 : si32 step 1 : si32 {
        x86_scf.yield
    }
    ret
}

// -----
// CHECK-LABEL:    x86_func.func @dyn_lb_static_ub(%lb: !x86.reg64<rcx>) {
//  CHECK-NEXT:      %step = x86.di.mov 2 : () -> !x86.reg64<rdx>
//  CHECK-NEXT:      %0 = x86.si.cmp %lb, 10 : (!x86.reg64<rcx>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jge %0 : !x86.rflags<rflags>, ^bb2(%lb : !x86.reg64<rcx>), ^bb1(%lb : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb1(%i: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86.label "scf_body_0_for"
//  CHECK-NEXT:      %i_1 = x86.rs.add %i, %step : (!x86.reg64<rcx>, !x86.reg64<rdx>) -> !x86.reg64<rcx>
//  CHECK-NEXT:      %1 = x86.si.cmp %i_1, 10 : (!x86.reg64<rcx>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jl %1 : !x86.rflags<rflags>, ^bb1(%i_1 : !x86.reg64<rcx>), ^bb2(%i_1 : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb2(%lb_end: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86.label "scf_body_end_0_for"
//  CHECK-NEXT:      x86_func.ret
//  CHECK-NEXT:    }
x86_func.func @dyn_lb_static_ub(%lb: !x86.reg64<rcx>) {
    %step = x86.di.mov 2 : () -> !x86.reg64<rdx>
    %lb_end = x86_scf.for %i : !x86.reg64<rcx> = %lb to 10 : si32 step %step {
        x86_scf.yield
    }
    ret
}

// -----
// CHECK-LABEL:    x86_func.func @dyn_lb_static_bounds(%lb: !x86.reg64<rcx>) {
//  CHECK-NEXT:      %0 = x86.si.cmp %lb, 12 : (!x86.reg64<rcx>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jge %0 : !x86.rflags<rflags>, ^bb2(%lb : !x86.reg64<rcx>), ^bb1(%lb : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb1(%i: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86.label "scf_body_0_for"
//  CHECK-NEXT:      %i_1 = x86.ri.add %i, 1 : (!x86.reg64<rcx>) -> !x86.reg64<rcx>
//  CHECK-NEXT:      %1 = x86.si.cmp %i_1, 12 : (!x86.reg64<rcx>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jl %1 : !x86.rflags<rflags>, ^bb1(%i_1 : !x86.reg64<rcx>), ^bb2(%i_1 : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb2(%lb_end: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86.label "scf_body_end_0_for"
//  CHECK-NEXT:      x86_func.ret
//  CHECK-NEXT:    }
x86_func.func @dyn_lb_static_bounds(%lb: !x86.reg64<rcx>) {
    %lb_end = x86_scf.for %i : !x86.reg64<rcx> = %lb to 12 : si32 step 1 : si32 {
        x86_scf.yield
    }
    ret
}

// -----
// CHECK-LABEL:    x86_func.func @dyn_ub_static_step() {
//  CHECK-NEXT:      %zero = x86.di.mov 0 : () -> !x86.reg64<rcx>
//  CHECK-NEXT:      %ub = x86.di.mov 20 : () -> !x86.reg64<r8>
//  CHECK-NEXT:      %0 = x86.ss.cmp %zero, %ub : (!x86.reg64<rcx>, !x86.reg64<r8>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jge %0 : !x86.rflags<rflags>, ^bb2(%zero : !x86.reg64<rcx>), ^bb1(%zero : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb1(%i: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86.label "scf_body_0_for"
//  CHECK-NEXT:      %i_1 = x86.ri.add %i, 3 : (!x86.reg64<rcx>) -> !x86.reg64<rcx>
//  CHECK-NEXT:      %1 = x86.ss.cmp %i_1, %ub : (!x86.reg64<rcx>, !x86.reg64<r8>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jl %1 : !x86.rflags<rflags>, ^bb1(%i_1 : !x86.reg64<rcx>), ^bb2(%i_1 : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb2(%zero_end: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86.label "scf_body_end_0_for"
//  CHECK-NEXT:      x86_func.ret
//  CHECK-NEXT:    }
x86_func.func @dyn_ub_static_step() {
    %zero = x86.di.mov 0 : () -> !x86.reg64<rcx>
    %ub = x86.di.mov 20 : () -> !x86.reg64<r8>
    %zero_end = x86_scf.for %i : !x86.reg64<rcx> = %zero to %ub step 3 : si32 {
        x86_scf.yield
    }
    ret
}

// -----
// CHECK-LABEL:    x86_func.func @known_empty() {
//  CHECK-NEXT:      %ten = x86.di.mov 10 : () -> !x86.reg64<rcx>
//  CHECK-NEXT:      %0 = x86.si.cmp %ten, 10 : (!x86.reg64<rcx>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jge %0 : !x86.rflags<rflags>, ^bb2(%ten : !x86.reg64<rcx>), ^bb1(%ten : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb1(%i: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86.label "scf_body_0_for"
//  CHECK-NEXT:      %i_1 = x86.ri.add %i, 1 : (!x86.reg64<rcx>) -> !x86.reg64<rcx>
//  CHECK-NEXT:      %1 = x86.si.cmp %i_1, 10 : (!x86.reg64<rcx>) -> !x86.rflags<rflags>
//  CHECK-NEXT:      x86.c.jl %1 : !x86.rflags<rflags>, ^bb1(%i_1 : !x86.reg64<rcx>), ^bb2(%i_1 : !x86.reg64<rcx>)
//  CHECK-NEXT:    ^bb2(%ten_end: !x86.reg64<rcx>):
//  CHECK-NEXT:      x86.label "scf_body_end_0_for"
//  CHECK-NEXT:      x86_func.ret
//  CHECK-NEXT:    }
x86_func.func @known_empty() {
    %ten = x86.di.mov 10 : () -> !x86.reg64<rcx>
    %ten_end = x86_scf.for %i : !x86.reg64<rcx> = %ten to 10 : si32 step 1 : si32 {
        x86_scf.yield
    }
    ret
}
