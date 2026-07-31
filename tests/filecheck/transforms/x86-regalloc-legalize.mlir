// RUN: xdsl-opt -p x86-regalloc-legalize %s --split-input-file --verify-diagnostics | filecheck %s

// L1: straight-line live inout.
x86_func.func @live_inout() {
  %a = x86.di.mov 1 : () -> !x86.reg64
  %imm = x86.di.mov 2 : () -> !x86.reg64
  %b = x86.rs.add %a, %imm : (!x86.reg64, !x86.reg64) -> !x86.reg64
  %c = x86.rs.add %a, %b : (!x86.reg64, !x86.reg64) -> !x86.reg64
  x86_func.ret
}

// CHECK-LABEL: @live_inout
// CHECK-NEXT:    %a = x86.di.mov 1 : () -> !x86.reg64
// CHECK-NEXT:    %imm = x86.di.mov 2 : () -> !x86.reg64
// CHECK-NEXT:    %a_1 = x86.ds.mov %a : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    %b = x86.rs.add %a_1, %imm : (!x86.reg64, !x86.reg64) -> !x86.reg64
// CHECK-NEXT:    %c = x86.rs.add %a, %b : (!x86.reg64, !x86.reg64) -> !x86.reg64
// CHECK-NEXT:    x86_func.ret

// -----

// L2: straight-line last use.
x86_func.func @last_use_inout() {
  %a = x86.di.mov 1 : () -> !x86.reg64
  %imm = x86.di.mov 2 : () -> !x86.reg64
  %b = x86.rs.add %a, %imm : (!x86.reg64, !x86.reg64) -> !x86.reg64
  x86_func.ret
}

// CHECK-LABEL: @last_use_inout
// CHECK-NEXT:    %a = x86.di.mov 1 : () -> !x86.reg64
// CHECK-NEXT:    %imm = x86.di.mov 2 : () -> !x86.reg64
// CHECK-NEXT:    %b = x86.rs.add %a, %imm : (!x86.reg64, !x86.reg64) -> !x86.reg64
// CHECK-NEXT:    x86_func.ret

// -----

// L3: live-in whose only use is the clobber (same as V2).
x86_func.func @for_outer_last_use_in_body(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: @for_outer_last_use_in_body
// CHECK:         x86_scf.for
// CHECK-NEXT:      %x_1 = x86.ds.mov %x : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      %x2 = x86.r.inc %x_1 : (!x86.reg64) -> !x86.reg64

// -----

// L4: body use before the clobber (V1).
x86_func.func @body_use_before_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    "test.op"(%x) : (!x86.reg64) -> ()
    %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: @body_use_before_clobber
// CHECK:           "test.op"(%x)
// CHECK-NEXT:      %x_1 = x86.ds.mov %x : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      %x2 = x86.r.inc %x_1 : (!x86.reg64) -> !x86.reg64

// -----

// L5: induction variable clobbered (V3).
x86_func.func @iv_clobbered(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    %i2 = x86.r.inc %i : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: @iv_clobbered
// CHECK:           %i_1 = x86.ds.mov %i : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      %i2 = x86.r.inc %i_1 : (!x86.reg64) -> !x86.reg64

// -----

// L6: clobber before the loop (V5).
x86_func.func @clobber_before_loop(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    "test.op"(%x) : (!x86.reg64) -> ()
  }
  x86_func.ret
}

// CHECK-LABEL: @clobber_before_loop
// CHECK:         %x_1 = x86.ds.mov %x : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    %x2 = x86.r.inc %x_1 : (!x86.reg64) -> !x86.reg64
// CHECK:         x86_scf.for
// CHECK-NEXT:      "test.op"(%x)

// -----

// L7: iter_args operand live after the loop (V6).
x86_func.func @iter_arg_live_after(%init: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %res = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %init) -> (!x86.reg64) {
    %a2 = x86.r.inc %a : (!x86.reg64) -> !x86.reg64
    x86_scf.yield %a2 : !x86.reg64
  }
  "test.op"(%init) : (!x86.reg64) -> ()
  x86_func.ret
}

// CHECK-LABEL: @iter_arg_live_after
// CHECK:         %init_1 = x86.ds.mov %init : (!x86.reg64) -> !x86.reg64
// CHECK:         iter_args(%a = %init_1)
// CHECK:         "test.op"(%init)

// -----

// L8: duplicate iter_args (V7).
x86_func.func @duplicate_iter_args(%v: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %r0, %r1 = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %v, %b = %v) -> (!x86.reg64, !x86.reg64) {
    x86_scf.yield %a, %b : !x86.reg64, !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: @duplicate_iter_args
// CHECK:         %v_1 = x86.ds.mov %v : (!x86.reg64) -> !x86.reg64
// CHECK:         iter_args(%a = %v_1, %b = %v)

// -----

// L9: nested loops (V8).
x86_func.func @nested_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    x86_scf.for %j : !x86.reg64 = %lb to %ub step %step {
      %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
    }
  }
  x86_func.ret
}

// CHECK-LABEL: @nested_clobber
// CHECK:         x86_scf.for
// CHECK:           x86_scf.for
// CHECK-NEXT:        %x_1 = x86.ds.mov %x : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        %x2 = x86.r.inc %x_1 : (!x86.reg64) -> !x86.reg64

// -----

// L10: accumulator pattern (V10) — no copy.
x86_func.func @iter_arg_accumulate(%init: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %res = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %init) -> (!x86.reg64) {
    "test.op"(%a) : (!x86.reg64) -> ()
    %a2 = x86.r.inc %a : (!x86.reg64) -> !x86.reg64
    x86_scf.yield %a2 : !x86.reg64
  }
  "test.op"(%res) : (!x86.reg64) -> ()
  x86_func.ret
}

// CHECK-LABEL: @iter_arg_accumulate
// CHECK-NOT:     x86.ds.mov

// -----

// L11: rof (V12).
x86_func.func @rof_body_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  x86_scf.rof %i : !x86.reg64 = %ub down to %lb step %step {
    %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// CHECK-LABEL: @rof_body_clobber
// CHECK:           %x_1 = x86.ds.mov %x : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      %x2 = x86.r.inc %x_1 : (!x86.reg64) -> !x86.reg64

// -----

// L12: duplicate iter_args, also live after the loop: both slots need a copy.
x86_func.func @duplicate_iter_args_live_after(%v: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %r0, %r1 = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %v, %b = %v) -> (!x86.reg64, !x86.reg64) {
    x86_scf.yield %a, %b : !x86.reg64, !x86.reg64
  }
  "test.op"(%v) : (!x86.reg64) -> ()
  x86_func.ret
}

// CHECK-LABEL: @duplicate_iter_args_live_after
// CHECK:         %v_1 = x86.ds.mov %v : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    %v_2 = x86.ds.mov %v : (!x86.reg64) -> !x86.reg64
// CHECK:         iter_args(%a = %v_1, %b = %v_2)
// CHECK:         "test.op"(%v)

// -----

// External / empty function declaration is a no-op.
x86_func.func @external(%ptr: !x86.reg64)

// CHECK-LABEL: @external

// -----

// CHECK: Cannot yet legalize func with multiple blocks.
x86_func.func @multi_block(%ptr: !x86.reg64) {
  %ptr2 = x86.r.inc %ptr : (!x86.reg64) -> !x86.reg64
  x86.fallthrough ^next()
^next:
  %ptr3 = x86.r.inc %ptr2 : (!x86.reg64) -> !x86.reg64
  x86_func.ret
}
