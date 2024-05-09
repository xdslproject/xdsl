// RUN: xdsl-opt -t x86-asm %s | filecheck %s

%0 = x86.get_register : () -> !x86.reg<rax>
%1 = x86.get_register : () -> !x86.reg<rdx>
%2 = x86.get_register : () -> !x86.reg<rcx>
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
%cmp = x86.rr.cmp %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.rflags<rflags>
// CHECK: cmp rax, rdx
x86.r.push %0 : (!x86.reg<rax>) -> ()
// CHECK: push rax
%pop, %poprsp = x86.r.pop %rsp : (!x86.reg<rsp>) -> (!x86.reg<rax>, !x86.reg<rsp>)
// CHECK: pop rax
%not = x86.r.not %0 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK: not rax

%r_idiv_rdx, %r_idiv_rax = x86.r.idiv %2, %1, %0 : (!x86.reg<rcx>, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK: idiv rcx

%rm_add_no_offset = x86.rm.add %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: add rax, [rdx]
%rm_add = x86.rm.add %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: add rax, [rdx+8]
%rm_sub = x86.rm.sub %0, %1, -8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: sub rax, [rdx-8]
%rm_imul = x86.rm.imul %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: imul rax, [rdx+8]
%rm_and = x86.rm.and %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: and rax, [rdx+8]
%rm_or = x86.rm.or %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: or rax, [rdx+8]
%rm_xor = x86.rm.xor %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: xor rax, [rdx+8]
%rm_mov = x86.rm.mov %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: mov rax, [rdx+8]
%rm_cmp = x86.rm.cmp %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.rflags<rflags>
// CHECK: cmp rax, [rdx+8]

%ri_add = x86.ri.add %0, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK: add rax, 2
%ri_sub = x86.ri.sub %0, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK: sub rax, 2
%ri_and = x86.ri.and %0, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK: and rax, 2
%ri_or = x86.ri.or %0, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK: or rax, 2
%ri_xor = x86.ri.xor %0, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK: xor rax, 2
%ri_mov = x86.ri.mov %0, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK: mov rax, 2

x86.mr.add %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK: add [rax], rdx
x86.mr.add %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK: add [rax+8], rdx
x86.mr.sub %0, %1, -8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK: sub [rax-8], rdx
x86.mr.and %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK: and [rax+8], rdx
x86.mr.or %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK: or [rax+8], rdx
x86.mr.xor %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK: xor [rax+8], rdx
x86.mr.mov %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK: mov [rax+8], rdx

x86.mi.add %0, 2 : (!x86.reg<rax>) -> ()
// CHECK: add [rax], 2
x86.mi.add %0, 2, 8 : (!x86.reg<rax>) -> ()
// CHECK: add [rax+8], 2
x86.mi.sub %0, 2, -8 : (!x86.reg<rax>) -> ()
// CHECK: sub [rax-8], 2
x86.mi.and %0, 2, 8 : (!x86.reg<rax>) -> ()
// CHECK: and [rax+8], 2
x86.mi.or %0, 2, 8 : (!x86.reg<rax>) -> ()
// CHECK: or [rax+8], 2
x86.mi.xor %0, 2, 8 : (!x86.reg<rax>) -> ()
// CHECK: xor [rax+8], 2
x86.mi.mov %0, 2, 8 : (!x86.reg<rax>) -> ()
// CHECK: mov [rax+8], 2

%rri_imul = x86.rri.imul %1, 2 : (!x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: imul rax, rdx, 2

%rmi_imul_no_offset = x86.rmi.imul %1, 2 : (!x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: imul rax, [rdx], 2
%rmi_imul = x86.rmi.imul %1, 2, 8 : (!x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: imul rax, [rdx+8], 2

x86.m.push %0 : (!x86.reg<rax>) -> ()
// CHECK: push [rax]
x86.m.push %0, 8 : (!x86.reg<rax>) -> ()
// CHECK: push [rax+8]
x86.m.neg %0 : (!x86.reg<rax>) -> ()
// CHECK: neg [rax]
x86.m.neg %0, 8 : (!x86.reg<rax>) -> ()
// CHECK: neg [rax+8]
x86.m.not %0, 8 : (!x86.reg<rax>) -> ()
// CHECK: not [rax+8]

x86.directive ".text"
// CHECK: .text
x86.directive ".align" "2"
// CHECK: .align 2
x86.label "label"
// CHECK: label:

func.func @funcyasm() {
    %3 = x86.get_register : () -> !x86.reg<rax>
    %4 = x86.get_register : () -> !x86.reg<rdx>
    %rflags = x86.rr.cmp %3, %4 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.rflags<rflags>
    // CHECK: cmp rax, rdx
    
    x86.s.jmp ^thenblock(%arg : !x86.reg<>)
    // CHECK: jmp then
    ^thenblock(%arg : !x86.reg<>):
    x86.label "then"
    // CHECK-NEXT: then:
    x86.s.je %rflags : !x86.rflags<rflags>, ^thenblock(%arg : !x86.reg<>), ^elseblock(%arg2 : !x86.reg<>)
    // CHECK-NEXT: je then
    ^elseblock(%arg2 : !x86.reg<>):
    x86.label "else"
    // CHECK-NEXT: else:

    x86.s.jmp ^thenblock(%arg : !x86.reg<>)
    // CHECK-NEXT: jmp then
}
