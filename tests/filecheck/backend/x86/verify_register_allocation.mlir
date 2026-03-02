// RUN: xdsl-opt -p verify-register-allocation --verify-diagnostics --split-input-file %s | filecheck %s

// CHECK-LABEL:    @inc
x86_func.func @inc(%ptr: !x86.reg<rax>) {
// CHECK-NEXT: %val = x86.dm.mov %ptr, 0 : (!x86.reg<rax>) -> !x86.reg<rcx>
// CHECK-NEXT: %ptr2 = x86.r.inc %ptr : (!x86.reg<rax>) -> !x86.reg<rax>
  %val = x86.dm.mov %ptr : (!x86.reg<rax>) -> !x86.reg<rcx>
  %ptr2 = x86.r.inc %ptr : (!x86.reg<rax>) -> !x86.reg<rax>

// CHECK-NEXT: x86_func.ret
  x86_func.ret
}

// -----

// CHECK-LABEL:    @inc0
x86_func.func @inc0(%ptr : !x86.reg) {
// CHECK-NEXT: %ptr2 = x86.r.inc %ptr : (!x86.reg) -> !x86.reg
// CHECK-NEXT: x86.fallthrough ^next()
// CHECK-NEXT: ^next:
// CHECK-NEXT: %ptr3 = x86.r.inc %ptr2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: x86_func.ret
  %ptr2 = x86.r.inc %ptr : (!x86.reg) -> !x86.reg
  x86.fallthrough ^next()
^next:
  %ptr3 = x86.r.inc %ptr2 : (!x86.reg) -> !x86.reg
  x86_func.ret
}

// -----

// CHECK: ptr should not be read after in/out usage
x86_func.func @inc2(%ptr : !x86.reg<rax>) {
  %ptr2 = x86.r.inc %ptr : (!x86.reg<rax>) -> !x86.reg<rax>
  %ptr3 = x86.r.inc %ptr : (!x86.reg<rax>) -> !x86.reg<rax>
  x86_func.ret
}

// -----

// CHECK: ptr should not be read after in/out usage
x86_func.func @inc3(%ptr : !x86.reg) {
  %ptr2 = x86.r.inc %ptr : (!x86.reg) -> !x86.reg
  %ptr3 = x86.r.inc %ptr : (!x86.reg) -> !x86.reg
  x86_func.ret
}

// -----

// CHECK: ptr should not be read after in/out usage
x86_func.func @inc4(%ptr : !x86.reg) {
  %init,%bound,%step = "test.op"(): () -> (!x86.reg,!x86.reg,!x86.reg)
  x86_scf.for %i : !x86.reg  = %init to %bound step %step {
    %ptr2 = x86.r.inc %ptr : (!x86.reg) -> !x86.reg
    %ptr3 = x86.r.inc %ptr : (!x86.reg) -> !x86.reg
  }
  x86_func.ret
}

// -----

// CHECK: ptr should not be read after in/out usage
x86_func.func @inc4(%ptr : !x86.reg) {
  %ptr2 = x86.r.inc %ptr : (!x86.reg) -> !x86.reg
  x86.fallthrough ^next()
^next:
  %ptr3 = x86.r.inc %ptr : (!x86.reg) -> !x86.reg
  x86_func.ret
}

// -----

// CHECK: ptr should not be read after in/out usage
x86_func.func @inc5(%ptr : !x86.reg) {
  x86.c.jmp ^bb2()
^bb1:
  x86.label "bb1"
  "test.op"(%ptr) : (!x86.reg) -> ()
  x86_func.ret
^bb2:
  x86.label "bb2"
  %ptr3 = x86.r.inc %ptr : (!x86.reg) -> !x86.reg
  x86.c.jmp ^bb1()
}

// -----

// CHECK-LABEL:    @inc6
x86_func.func @inc6(%ptr: !x86.reg<rax>)
