// RUN: xdsl-opt -p x86-allocate-registers --verify-diagnostics --split-input-file %s | filecheck %s

// CHECK-LABEL:    @inc
x86_func.func @inc(%ptr: !x86.reg) {
// CHECK-NEXT: %val = x86.dm.mov %ptr, 0 : (!x86.reg<rax>) -> !x86.reg<rcx>
// CHECK-NEXT: %ptr2 = x86.r.inc %ptr : (!x86.reg<rax>) -> !x86.reg<rax>
  %val = x86.dm.mov %ptr : (!x86.reg) -> !x86.reg
  %ptr2 = x86.r.inc %ptr : (!x86.reg) -> !x86.reg

// CHECK-NEXT: x86_func.ret
  x86_func.ret
}

// -----

x86_func.func @inc2(%ptr: !x86.reg) {
  %ptr2 = x86.r.inc %ptr : (!x86.reg) -> !x86.reg
  %ptr3 = x86.r.inc %ptr : (!x86.reg) -> !x86.reg
  // expected-error {{Inout register operand at index 0 used after write.}}
  x86_func.ret
}
