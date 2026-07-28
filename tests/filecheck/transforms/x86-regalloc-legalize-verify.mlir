// RUN: xdsl-opt -p x86-regalloc-legalize,x86-regalloc-verify-liveness %s --split-input-file | filecheck %s

// Straight-line live inout.
x86_func.func @live_inout() {
  %a = x86.di.mov 1 : () -> !x86.reg64
  %imm = x86.di.mov 2 : () -> !x86.reg64
  %b = x86.rs.add %a, %imm : (!x86.reg64, !x86.reg64) -> !x86.reg64
  %c = x86.rs.add %a, %b : (!x86.reg64, !x86.reg64) -> !x86.reg64
  x86_func.ret
}

// CHECK-LABEL: @live_inout
// CHECK:         %a_1 = x86.ds.mov %a
// CHECK:         %b = x86.rs.add %a_1, %imm
// CHECK:         %c = x86.rs.add %a, %b

// -----

// V1: body use before clobber.
x86_func.func @body_use_before_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    "test.op"(%x) : (!x86.reg64) -> ()
    %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: @body_use_before_clobber
// CHECK:         x86.ds.mov %x

// -----

// V2: live-in only use is clobber.
x86_func.func @body_only_use_is_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: @body_only_use_is_clobber
// CHECK:         x86.ds.mov %x

// -----

// V3: IV clobbered.
x86_func.func @iv_clobbered(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    %i2 = x86.r.inc %i : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: @iv_clobbered
// CHECK:         x86.ds.mov %i

// -----

// V4: ub clobbered.
x86_func.func @ub_clobbered(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    %u2 = x86.r.inc %ub : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: @ub_clobbered
// CHECK:         x86.ds.mov %ub

// -----

// V5: clobber before loop.
x86_func.func @clobber_before_loop(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    "test.op"(%x) : (!x86.reg64) -> ()
  }
  x86_func.ret
}

// CHECK-LABEL: @clobber_before_loop
// CHECK:         x86.ds.mov %x

// -----

// V6: iter_arg live after loop.
x86_func.func @iter_arg_live_after(%init: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %res = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %init) -> (!x86.reg64) {
    %a2 = x86.r.inc %a : (!x86.reg64) -> !x86.reg64
    x86_scf.yield %a2 : !x86.reg64
  }
  "test.op"(%init) : (!x86.reg64) -> ()
  x86_func.ret
}

// CHECK-LABEL: @iter_arg_live_after
// CHECK:         x86.ds.mov %init

// -----

// V7: duplicate iter_args.
x86_func.func @duplicate_iter_args(%v: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %r0, %r1 = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %v, %b = %v) -> (!x86.reg64, !x86.reg64) {
    x86_scf.yield %a, %b : !x86.reg64, !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: @duplicate_iter_args
// CHECK:         x86.ds.mov %v

// -----

// Duplicate iter_args, also live after the loop.
x86_func.func @duplicate_iter_args_live_after(%v: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %r0, %r1 = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %v, %b = %v) -> (!x86.reg64, !x86.reg64) {
    x86_scf.yield %a, %b : !x86.reg64, !x86.reg64
  }
  "test.op"(%v) : (!x86.reg64) -> ()
  x86_func.ret
}

// CHECK-LABEL: @duplicate_iter_args_live_after
// CHECK:         iter_args(%a = %v_1, %b = %v_2)

// -----

// V8: nested clobber.
x86_func.func @nested_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    x86_scf.for %j : !x86.reg64 = %lb to %ub step %step {
      %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
    }
  }
  x86_func.ret
}

// CHECK-LABEL: @nested_clobber
// CHECK:         x86.ds.mov %x

// -----

// V11: iter_arg read after clobber in body.
x86_func.func @iter_arg_read_after_clobber(%init: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %res = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %init) -> (!x86.reg64) {
    %a2 = x86.r.inc %a : (!x86.reg64) -> !x86.reg64
    "test.op"(%a) : (!x86.reg64) -> ()
    x86_scf.yield %a2 : !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: @iter_arg_read_after_clobber
// CHECK:         x86.ds.mov %a

// -----

// V12: rof.
x86_func.func @rof_body_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  x86_scf.rof %i : !x86.reg64 = %ub down to %lb step %step {
    %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: @rof_body_clobber
// CHECK:         x86.ds.mov %x
