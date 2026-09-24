// RUN: xdsl-opt -p x86-regalloc-legalize,x86-regalloc-verify-liveness %s --split-input-file | filecheck %s

// Straight-line live inout.
x86_func.func @live_inout() {
  %a = x86.di.mov 1 : () -> !x86.reg64
  %imm = x86.di.mov 2 : () -> !x86.reg64
  %b = x86.rs.add %a, %imm : (!x86.reg64, !x86.reg64) -> !x86.reg64
  %c = x86.rs.add %a, %b : (!x86.reg64, !x86.reg64) -> !x86.reg64
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @live_inout() {
// CHECK-NEXT:    %a = x86.di.mov 1 : () -> !x86.reg64
// CHECK-NEXT:    %imm = x86.di.mov 2 : () -> !x86.reg64
// CHECK-NEXT:    %a_1 = x86.ds.mov %a : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    %b = x86.rs.add %a_1, %imm : (!x86.reg64, !x86.reg64) -> !x86.reg64
// CHECK-NEXT:    %c = x86.rs.add %a, %b : (!x86.reg64, !x86.reg64) -> !x86.reg64
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// V1: body use before clobber.
x86_func.func @body_use_before_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    "test.op"(%x) : (!x86.reg64) -> ()
    %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @body_use_before_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
// CHECK-NEXT:    %lb_end = x86_scf.for %i : !x86.reg64  = %lb to %ub step %step {
// CHECK-NEXT:      "test.op"(%x) : (!x86.reg64) -> ()
// CHECK-NEXT:      %x_1 = x86.ds.mov %x : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      %x2 = x86.r.inc %x_1 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    }
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// V2: live-in only use is clobber.
x86_func.func @body_only_use_is_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @body_only_use_is_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
// CHECK-NEXT:    %lb_end = x86_scf.for %i : !x86.reg64  = %lb to %ub step %step {
// CHECK-NEXT:      %x_1 = x86.ds.mov %x : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      %x2 = x86.r.inc %x_1 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    }
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// V3: IV clobbered.
x86_func.func @iv_clobbered(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    %i2 = x86.r.inc %i : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @iv_clobbered(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
// CHECK-NEXT:    %lb_end = x86_scf.for %i : !x86.reg64  = %lb to %ub step %step {
// CHECK-NEXT:      %i_1 = x86.ds.mov %i : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      %i2 = x86.r.inc %i_1 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    }
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// V4: ub clobbered.
x86_func.func @ub_clobbered(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    %u2 = x86.r.inc %ub : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @ub_clobbered(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
// CHECK-NEXT:    %lb_end = x86_scf.for %i : !x86.reg64  = %lb to %ub step %step {
// CHECK-NEXT:      %ub_1 = x86.ds.mov %ub : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      %u2 = x86.r.inc %ub_1 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    }
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// V5: clobber before loop.
x86_func.func @clobber_before_loop(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    "test.op"(%x) : (!x86.reg64) -> ()
  }
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @clobber_before_loop(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
// CHECK-NEXT:    %x_1 = x86.ds.mov %x : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    %x2 = x86.r.inc %x_1 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    %lb_end = x86_scf.for %i : !x86.reg64  = %lb to %ub step %step {
// CHECK-NEXT:      "test.op"(%x) : (!x86.reg64) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// V6: iter_arg live after loop.
x86_func.func @iter_arg_live_after(%init: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end, %res = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %init) -> (!x86.reg64) {
    %a2 = x86.r.inc %a : (!x86.reg64) -> !x86.reg64
    x86_scf.yield %a2 : !x86.reg64
  }
  "test.op"(%init) : (!x86.reg64) -> ()
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @iter_arg_live_after(%init: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
// CHECK-NEXT:    %init_1 = x86.ds.mov %init : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    %lb_end, %res = x86_scf.for %i : !x86.reg64  = %lb to %ub step %step iter_args(%a = %init_1) -> (!x86.reg64) {
// CHECK-NEXT:      %a2 = x86.r.inc %a : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      x86_scf.yield %a2 : !x86.reg64
// CHECK-NEXT:    }
// CHECK-NEXT:    "test.op"(%init) : (!x86.reg64) -> ()
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// V7: duplicate iter_args.
x86_func.func @duplicate_iter_args(%v: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end, %r0, %r1 = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %v, %b = %v) -> (!x86.reg64, !x86.reg64) {
    x86_scf.yield %a, %b : !x86.reg64, !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @duplicate_iter_args(%v: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
// CHECK-NEXT:    %v_1 = x86.ds.mov %v : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    %lb_end, %r0, %r1 = x86_scf.for %i : !x86.reg64  = %lb to %ub step %step iter_args(%a = %v_1, %b = %v) -> (!x86.reg64, !x86.reg64) {
// CHECK-NEXT:      x86_scf.yield %a, %b : !x86.reg64, !x86.reg64
// CHECK-NEXT:    }
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// Duplicate iter_args, also live after the loop.
x86_func.func @duplicate_iter_args_live_after(%v: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end, %r0, %r1 = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %v, %b = %v) -> (!x86.reg64, !x86.reg64) {
    x86_scf.yield %a, %b : !x86.reg64, !x86.reg64
  }
  "test.op"(%v) : (!x86.reg64) -> ()
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @duplicate_iter_args_live_after(%v: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
// CHECK-NEXT:    %v_1 = x86.ds.mov %v : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    %v_2 = x86.ds.mov %v : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    %lb_end, %r0, %r1 = x86_scf.for %i : !x86.reg64  = %lb to %ub step %step iter_args(%a = %v_1, %b = %v_2) -> (!x86.reg64, !x86.reg64) {
// CHECK-NEXT:      x86_scf.yield %a, %b : !x86.reg64, !x86.reg64
// CHECK-NEXT:    }
// CHECK-NEXT:    "test.op"(%v) : (!x86.reg64) -> ()
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// V8: nested clobber.
x86_func.func @nested_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    %lb_end_1 = x86_scf.for %j : !x86.reg64 = %lb to %ub step %step {
      %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
    }
  }
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @nested_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
// CHECK-NEXT:    %lb_1 = x86.ds.mov %lb : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    %lb_end = x86_scf.for %i : !x86.reg64  = %lb_1 to %ub step %step {
// CHECK-NEXT:      %lb_2 = x86.ds.mov %lb : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      %lb_end_1 = x86_scf.for %j : !x86.reg64  = %lb_2 to %ub step %step {
// CHECK-NEXT:        %x_1 = x86.ds.mov %x : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        %x2 = x86.r.inc %x_1 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// V11: iter_arg read after clobber in body.
x86_func.func @iter_arg_read_after_clobber(%init: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end, %res = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %init) -> (!x86.reg64) {
    %a2 = x86.r.inc %a : (!x86.reg64) -> !x86.reg64
    "test.op"(%a) : (!x86.reg64) -> ()
    x86_scf.yield %a2 : !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @iter_arg_read_after_clobber(%init: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
// CHECK-NEXT:    %lb_end, %res = x86_scf.for %i : !x86.reg64  = %lb to %ub step %step iter_args(%a = %init) -> (!x86.reg64) {
// CHECK-NEXT:      %a_1 = x86.ds.mov %a : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      %a2 = x86.r.inc %a_1 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      "test.op"(%a) : (!x86.reg64) -> ()
// CHECK-NEXT:      x86_scf.yield %a2 : !x86.reg64
// CHECK-NEXT:    }
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// V12: rof.
x86_func.func @rof_body_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.rof %i : !x86.reg64 = %ub down to %lb step %step {
    %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @rof_body_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
// CHECK-NEXT:    %lb_end = x86_scf.rof %i : !x86.reg64  = %ub down  to %lb step %step {
// CHECK-NEXT:      %x_1 = x86.ds.mov %x : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      %x2 = x86.r.inc %x_1 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    }
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// lb live after loop.
x86_func.func @lb_live_after(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    x86_scf.yield
  }
  "test.op"(%lb) : (!x86.reg64) -> ()
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @lb_live_after(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
// CHECK-NEXT:    %lb_1 = x86.ds.mov %lb : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    %lb_end = x86_scf.for %i : !x86.reg64  = %lb_1 to %ub step %step {
// CHECK-NEXT:    }
// CHECK-NEXT:    "test.op"(%lb) : (!x86.reg64) -> ()
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// Preserve the initial (upper) bound if it is read after the reverse loop.
x86_func.func @start_live_after(%lb: !x86.reg64, %ub: !x86.reg64) {
  %iv_end = x86_scf.rof %i : !x86.reg64 = %ub down to %lb step 1 : si32 {
  }
  "test.op"(%ub) : (!x86.reg64) -> ()
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @start_live_after(%lb: !x86.reg64, %ub: !x86.reg64) {
// CHECK-NEXT:    %ub_1 = x86.ds.mov %ub : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    %iv_end = x86_scf.rof %i : !x86.reg64  = %ub_1 down  to %lb step 1 : si32 {
// CHECK-NEXT:    }
// CHECK-NEXT:    "test.op"(%ub) : (!x86.reg64) -> ()
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// The termination (lower) bound is read-only, so it does not need a copy.
x86_func.func @stop_live_after(%lb: !x86.reg64, %ub: !x86.reg64) {
  %iv_end = x86_scf.rof %i : !x86.reg64 = %ub down to %lb step 1 : si32 {
  }
  "test.op"(%lb) : (!x86.reg64) -> ()
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @stop_live_after(%lb: !x86.reg64, %ub: !x86.reg64) {
// CHECK-NEXT:    %iv_end = x86_scf.rof %i : !x86.reg64  = %ub down  to %lb step 1 : si32 {
// CHECK-NEXT:    }
// CHECK-NEXT:    "test.op"(%lb) : (!x86.reg64) -> ()
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// A step that aliases the initial bound must survive the implicit back-edge use,
// even with an empty body and no uses after the loop.
x86_func.func @start_is_step(%lb: !x86.reg64, %ub: !x86.reg64) {
  %iv_end = x86_scf.rof %i : !x86.reg64 = %ub down to %lb step %ub {
  }
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @start_is_step(%lb: !x86.reg64, %ub: !x86.reg64) {
// CHECK-NEXT:    %ub_1 = x86.ds.mov %ub : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    %iv_end = x86_scf.rof %i : !x86.reg64  = %ub_1 down  to %lb step %ub {
// CHECK-NEXT:    }
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// A nested loop must not clobber the outer termination bound.
x86_func.func @nested_rof(%lb: !x86.reg64, %ub: !x86.reg64, %inner_lb: !x86.reg64) {
  %iv_end = x86_scf.rof %i : !x86.reg64 = %ub down to %lb step 1 : si32 {
    %inner_end = x86_scf.rof %j : !x86.reg64 = %lb down to %inner_lb step 1 : si32 {
    }
  }
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @nested_rof(%lb: !x86.reg64, %ub: !x86.reg64, %inner_lb: !x86.reg64) {
// CHECK-NEXT:    %iv_end = x86_scf.rof %i : !x86.reg64  = %ub down  to %lb step 1 : si32 {
// CHECK-NEXT:      %lb_1 = x86.ds.mov %lb : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      %inner_end = x86_scf.rof %j : !x86.reg64  = %lb_1 down  to %inner_lb step 1 : si32 {
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }
