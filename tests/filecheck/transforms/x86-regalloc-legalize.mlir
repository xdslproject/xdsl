// RUN: xdsl-opt -p x86-regalloc-legalize %s --split-input-file --verify-diagnostics | filecheck %s

// L1: straight-line live inout.
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

// L2: straight-line last use.
x86_func.func @last_use_inout() {
  %a = x86.di.mov 1 : () -> !x86.reg64
  %imm = x86.di.mov 2 : () -> !x86.reg64
  %b = x86.rs.add %a, %imm : (!x86.reg64, !x86.reg64) -> !x86.reg64
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @last_use_inout() {
// CHECK-NEXT:    %a = x86.di.mov 1 : () -> !x86.reg64
// CHECK-NEXT:    %imm = x86.di.mov 2 : () -> !x86.reg64
// CHECK-NEXT:    %b = x86.rs.add %a, %imm : (!x86.reg64, !x86.reg64) -> !x86.reg64
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// L3: live-in whose only use is the clobber (same as V2).
x86_func.func @for_outer_last_use_in_body(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @for_outer_last_use_in_body(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
// CHECK-NEXT:    %lb_end = x86_scf.for %i : !x86.reg64  = %lb to %ub step %step {
// CHECK-NEXT:      %x_1 = x86.ds.mov %x : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      %x2 = x86.r.inc %x_1 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    }
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// L4: body use before the clobber (V1).
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

// L5: induction variable clobbered (V3).
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

// L6: clobber before the loop (V5).
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

// L7: iter_args operand live after the loop (V6).
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

// L8: duplicate iter_args (V7).
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

// L9: nested loops (V8).
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

// L10: accumulator pattern (V10) — no copy.
x86_func.func @iter_arg_accumulate(%init: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end, %res = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %init) -> (!x86.reg64) {
    "test.op"(%a) : (!x86.reg64) -> ()
    %a2 = x86.r.inc %a : (!x86.reg64) -> !x86.reg64
    x86_scf.yield %a2 : !x86.reg64
  }
  "test.op"(%res) : (!x86.reg64) -> ()
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @iter_arg_accumulate(%init: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
// CHECK-NEXT:    %lb_end, %res = x86_scf.for %i : !x86.reg64  = %lb to %ub step %step iter_args(%a = %init) -> (!x86.reg64) {
// CHECK-NEXT:      "test.op"(%a) : (!x86.reg64) -> ()
// CHECK-NEXT:      %a2 = x86.r.inc %a : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      x86_scf.yield %a2 : !x86.reg64
// CHECK-NEXT:    }
// CHECK-NEXT:    "test.op"(%res) : (!x86.reg64) -> ()
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// L11: rof (V12).
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

// L12: duplicate iter_args, also live after the loop: both slots need a copy.
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

// L13: lb live after the loop — copy before for; original lb still used after.
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

// L14: lb last use — no copy.
x86_func.func @lb_last_use(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    x86_scf.yield
  }
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @lb_last_use(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
// CHECK-NEXT:    %lb_end = x86_scf.for %i : !x86.reg64  = %lb to %ub step %step {
// CHECK-NEXT:    }
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// -----

// External / empty function declaration is a no-op.
x86_func.func @external(%ptr: !x86.reg64)

// CHECK-LABEL: x86_func.func @external(!x86.reg64) -> ()

// -----

// CHECK: Cannot yet legalize func with multiple blocks.
x86_func.func @multi_block(%ptr: !x86.reg64) {
  %ptr2 = x86.r.inc %ptr : (!x86.reg64) -> !x86.reg64
  x86.fallthrough ^next()
^next:
  %ptr3 = x86.r.inc %ptr2 : (!x86.reg64) -> !x86.reg64
  x86_func.ret
}
