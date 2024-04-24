// RUN: xdsl-opt -t x86-asm %s | filecheck %s

%0 = x86.get_register : () -> !x86.reg<rax>
%1 = x86.get_register : () -> !x86.reg<rdx>
%rsp = x86.get_register : () -> !x86.reg<rsp>

%add = x86.rr.add %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: add rax, rdx
%sub = x86.rr.sub %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: sub rax, rdx
%imul = x86.rr.imul %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: imul rax, rdx
%and = x86.rr.and %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: and rax, rdx
%or = x86.rr.or %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: or rax, rdx
%xor = x86.rr.xor %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: xor rax, rdx
%mov = x86.rr.mov %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: mov rax, rdx
x86.r.push %0 : (!x86.reg<rax>) -> ()
// CHECK: push rax
%pop, %poprsp = x86.r.pop %rsp : (!x86.reg<rsp>) -> (!x86.reg<rax>, !x86.reg<rsp>)
// CHECK: pop rax
%not = x86.r.not %0 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK: not rax
