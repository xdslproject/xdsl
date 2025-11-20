// RUN: xdsl-opt -p x86-verify-register-allocation --verify-diagnostics --split-input-file %s | filecheck %s

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

// CHECK: ptr should not be read after in/out usage
x86_func.func @inc2(%ptr : !x86.reg<rax>) {
  %ptr2 = x86.r.inc %ptr : (!x86.reg<rax>) -> !x86.reg<rax>
  %ptr3 = x86.r.inc %ptr : (!x86.reg<rax>) -> !x86.reg<rax>
  x86_func.ret
}
