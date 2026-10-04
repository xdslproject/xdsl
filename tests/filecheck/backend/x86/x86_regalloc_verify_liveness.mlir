// RUN: xdsl-opt -p x86-regalloc-verify-liveness --verify-diagnostics --split-input-file %s | filecheck %s

// CHECK-LABEL:    @inc
x86_func.func @inc(%ptr: !x86.reg64<rax>) {
// CHECK-NEXT: %val = x86.dm.mov [%ptr] : (!x86.reg64<rax>) -> !x86.reg64<rcx>
// CHECK-NEXT: %ptr2 = x86.r.inc %ptr : (!x86.reg64<rax>) -> !x86.reg64<rax>
  %val = x86.dm.mov [%ptr] : (!x86.reg64<rax>) -> !x86.reg64<rcx>
  %ptr2 = x86.r.inc %ptr : (!x86.reg64<rax>) -> !x86.reg64<rax>

// CHECK-NEXT: x86_func.ret
  x86_func.ret
}


// -----

// CHECK: lb should not be read after in/out usage
x86_func.func @lb_live_after(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    x86_scf.yield
  }
  "test.op"(%lb) : (!x86.reg64) -> ()
  x86_func.ret
}

// -----

// CHECK: Cannot yet verify register liveness for regions with multiple blocks.
x86_func.func @inc0(%ptr: !x86.reg64) {
  %ptr2 = x86.r.inc %ptr : (!x86.reg64) -> !x86.reg64
  x86.fallthrough ^next()
^next:
  %ptr3 = x86.r.inc %ptr2 : (!x86.reg64) -> !x86.reg64
  x86_func.ret
}

// -----

// CHECK: ptr should not be read after in/out usage
x86_func.func @inc2(%ptr: !x86.reg64<rax>) {
  %ptr2 = x86.r.inc %ptr : (!x86.reg64<rax>) -> !x86.reg64<rax>
  %ptr3 = x86.r.inc %ptr : (!x86.reg64<rax>) -> !x86.reg64<rax>
  x86_func.ret
}

// -----

// CHECK: ptr should not be read after in/out usage
x86_func.func @inc3(%ptr: !x86.reg64) {
  %ptr2 = x86.r.inc %ptr : (!x86.reg64) -> !x86.reg64
  %ptr3 = x86.r.inc %ptr : (!x86.reg64) -> !x86.reg64
  x86_func.ret
}

// -----

// CHECK: ptr should not be read after in/out usage
x86_func.func @inc4(%ptr: !x86.reg64) {
  %init,%bound,%step = "test.op"(): () -> (!x86.reg64,!x86.reg64,!x86.reg64)
  %init_1 = x86_scf.for %i : !x86.reg64  = %init to %bound step %step {
    %ptr2 = x86.r.inc %ptr : (!x86.reg64) -> !x86.reg64
    %ptr3 = x86.r.inc %ptr : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}


// -----

// CHECK: lb should not be read after in/out usage
x86_func.func @lb_live_after(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    x86_scf.yield
  }
  "test.op"(%lb) : (!x86.reg64) -> ()
  x86_func.ret
}

// -----

// CHECK: Cannot yet verify register liveness for regions with multiple blocks.
x86_func.func @inc4(%ptr: !x86.reg64) {
  %ptr2 = x86.r.inc %ptr : (!x86.reg64) -> !x86.reg64
  x86.fallthrough ^next()
^next:
  %ptr3 = x86.r.inc %ptr : (!x86.reg64) -> !x86.reg64
  x86_func.ret
}


// -----

// CHECK: lb should not be read after in/out usage
x86_func.func @lb_live_after(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    x86_scf.yield
  }
  "test.op"(%lb) : (!x86.reg64) -> ()
  x86_func.ret
}

// -----

// CHECK: Cannot yet verify register liveness for regions with multiple blocks.
x86_func.func @inc5(%ptr: !x86.reg64) {
  x86.c.jmp ^bb2()
^bb1:
  x86.label "bb1"
  "test.op"(%ptr) : (!x86.reg64) -> ()
  x86_func.ret
^bb2:
  x86.label "bb2"
  %ptr3 = x86.r.inc %ptr : (!x86.reg64) -> !x86.reg64
  x86.c.jmp ^bb1()
}

// -----

// CHECK-LABEL:    @inc6
x86_func.func @inc6(%ptr: !x86.reg64<rax>)

// -----

// CHECK: x should not be read after in/out usage
x86_func.func @body_use_before_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    "test.op"(%x) : (!x86.reg64) -> ()
    %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// -----

// CHECK: x should not be read after in/out usage
x86_func.func @body_only_use_is_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// -----

// CHECK: i should not be read after in/out usage
x86_func.func @iv_clobbered(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    %i2 = x86.r.inc %i : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// -----

// CHECK: ub should not be read after in/out usage
x86_func.func @ub_clobbered(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    %u2 = x86.r.inc %ub : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// -----

// CHECK: x should not be read after in/out usage
x86_func.func @clobber_before_loop(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    "test.op"(%x) : (!x86.reg64) -> ()
  }
  x86_func.ret
}

// -----

// CHECK: init should not be read after in/out usage
x86_func.func @iter_arg_live_after(%init: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end, %res = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %init) -> (!x86.reg64) {
    %a2 = x86.r.inc %a : (!x86.reg64) -> !x86.reg64
    x86_scf.yield %a2 : !x86.reg64
  }
  "test.op"(%init) : (!x86.reg64) -> ()
  x86_func.ret
}

// -----

// CHECK: Value %v is used by more than one in/out operand
x86_func.func @duplicate_iter_args(%v: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end, %r0, %r1 = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %v, %b = %v) -> (!x86.reg64, !x86.reg64) {
    x86_scf.yield %a, %b : !x86.reg64, !x86.reg64
  }
  x86_func.ret
}

// -----

// CHECK: Value is used by more than one in/out operand
x86_func.func @duplicate_iter_args(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %0 = x86.get_register : !x86.reg64
  %lb_end, %r0, %r1 = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %0, %b = %0) -> (!x86.reg64, !x86.reg64) {
    x86_scf.yield %a, %b : !x86.reg64, !x86.reg64
  }
  x86_func.ret
}

// -----

// CHECK: x should not be read after in/out usage
x86_func.func @nested_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    %lb_end_1 = x86_scf.for %j : !x86.reg64 = %lb to %ub step %step {
      %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
    }
  }
  x86_func.ret
}

// -----

// CHECK-LABEL: @loop_live_in_untouched
x86_func.func @loop_live_in_untouched(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    "test.op"(%x) : (!x86.reg64) -> ()
  }
  x86_func.ret
}

// -----

// CHECK-LABEL: @iter_arg_accumulate
x86_func.func @iter_arg_accumulate(%init: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end, %res = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %init) -> (!x86.reg64) {
    "test.op"(%a) : (!x86.reg64) -> ()
    %a2 = x86.r.inc %a : (!x86.reg64) -> !x86.reg64
    x86_scf.yield %a2 : !x86.reg64
  }
  "test.op"(%res) : (!x86.reg64) -> ()
  x86_func.ret
}

// -----

// CHECK: a should not be read after in/out usage
x86_func.func @iter_arg_read_after_clobber(%init: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end, %res = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%a = %init) -> (!x86.reg64) {
    %a2 = x86.r.inc %a : (!x86.reg64) -> !x86.reg64
    "test.op"(%a) : (!x86.reg64) -> ()
    x86_scf.yield %a2 : !x86.reg64
  }
  x86_func.ret
}

// -----

// CHECK: x should not be read after in/out usage
x86_func.func @rof_body_clobber(%x: !x86.reg64, %lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.rof %i : !x86.reg64 = %ub down to %lb step %step {
    %x2 = x86.r.inc %x : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}


// -----

// CHECK: lb should not be read after in/out usage
x86_func.func @lb_live_after(%lb: !x86.reg64, %ub: !x86.reg64, %step: !x86.reg64) {
  %lb_end = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step {
    x86_scf.yield
  }
  "test.op"(%lb) : (!x86.reg64) -> ()
  x86_func.ret
}

// -----

// CHECK: Cannot yet verify register liveness for regions with multiple blocks.
x86_func.func @multi_block(%ptr: !x86.reg64) {
  %ptr2 = x86.r.inc %ptr : (!x86.reg64) -> !x86.reg64
  x86.fallthrough ^next()
^next:
  %ptr3 = x86.r.inc %ptr2 : (!x86.reg64) -> !x86.reg64
  x86_func.ret
}

// -----

// CHECK: Cannot verify register liveness through test.op
x86_func.func @unsupported_region(%x: !x86.reg64) {
  "test.op"() ({
    "test.termop"() : () -> ()
  }) : () -> ()
  x86_func.ret
}
