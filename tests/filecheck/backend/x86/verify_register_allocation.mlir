// RUN: xdsl-opt -p verify-register-allocation --verify-diagnostics --split-input-file %s | filecheck %s

// CHECK-LABEL:    @inc
x86_func.func @inc(%ptr: !x86.reg64<rax>) {
// CHECK-NEXT: %val = x86.dm.mov %ptr : (!x86.reg64<rax>) -> !x86.reg64<rcx>
// CHECK-NEXT: %ptr2 = x86.r.inc %ptr : (!x86.reg64<rax>) -> !x86.reg64<rax>
  %val = x86.dm.mov %ptr : (!x86.reg64<rax>) -> !x86.reg64<rcx>
  %ptr2 = x86.r.inc %ptr : (!x86.reg64<rax>) -> !x86.reg64<rax>

// CHECK-NEXT: x86_func.ret
  x86_func.ret
}

// -----

// CHECK-LABEL:    @inc0
x86_func.func @inc0(%ptr: !x86.reg64) {
// CHECK-NEXT: %ptr2 = x86.r.inc %ptr : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT: x86.fallthrough ^next()
// CHECK-NEXT: ^next:
// CHECK-NEXT: %ptr3 = x86.r.inc %ptr2 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT: x86_func.ret
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
  x86_scf.for %i : !x86.reg64  = %init to %bound step %step {
    %ptr2 = x86.r.inc %ptr : (!x86.reg64) -> !x86.reg64
    %ptr3 = x86.r.inc %ptr : (!x86.reg64) -> !x86.reg64
  }
  x86_func.ret
}

// -----

// CHECK: ptr should not be read after in/out usage
x86_func.func @inc4(%ptr: !x86.reg64) {
  %ptr2 = x86.r.inc %ptr : (!x86.reg64) -> !x86.reg64
  x86.fallthrough ^next()
^next:
  %ptr3 = x86.r.inc %ptr : (!x86.reg64) -> !x86.reg64
  x86_func.ret
}

// -----

// CHECK: ptr should not be read after in/out usage
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
