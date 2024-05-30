// RUN: xdsl-opt -t x86-asm %s | filecheck %s

%0 = x86.get_register : () -> !x86.reg<rax>
%1 = x86.get_register : () -> !x86.reg<rdx>
%2 = x86.get_register : () -> !x86.reg<rcx>
%rsp = x86.get_register : () -> !x86.reg<rsp>

%rr_add = x86.rr.add %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK: add rax, rdx
%rr_sub = x86.rr.sub %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: sub rax, rdx
%rr_imul = x86.rr.imul %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: imul rax, rdx
%rr_and = x86.rr.and %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: and rax, rdx
%rr_or = x86.rr.or %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: or rax, rdx
%rr_xor = x86.rr.xor %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: xor rax, rdx
%rr_mov = x86.rr.mov %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: mov rax, rdx
%rr_cmp = x86.rr.cmp %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.rflags<rflags>
// CHECK-NEXT: cmp rax, rdx
%r_pushrsp = x86.r.push %rsp, %0 : (!x86.reg<rsp>, !x86.reg<rax>) -> !x86.reg<rsp>
// CHECK-NEXT: push rax
%r_pop, %r_poprsp = x86.r.pop %rsp : (!x86.reg<rsp>) -> (!x86.reg<rax>, !x86.reg<rsp>)
// CHECK-NEXT: pop rax
%r_neg = x86.r.neg %0 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: neg rax
%r_not = x86.r.not %0 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: not rax
%r_inc = x86.r.inc %0 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: inc rax
%r_dec = x86.r.dec %0 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: dec rax

%r_idiv_rdx, %r_idiv_rax = x86.r.idiv %2, %1, %0 : (!x86.reg<rcx>, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: idiv rcx
%r_imul_rdx, %r_imul_rax = x86.r.imul %2, %0 : (!x86.reg<rcx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: imul rcx

%rm_add_no_offset = x86.rm.add %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: add rax, [rdx]
%rm_add = x86.rm.add %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: add rax, [rdx+8]
%rm_sub = x86.rm.sub %0, %1, -8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: sub rax, [rdx-8]
%rm_imul = x86.rm.imul %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: imul rax, [rdx+8]
%rm_and = x86.rm.and %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: and rax, [rdx+8]
%rm_or = x86.rm.or %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: or rax, [rdx+8]
%rm_xor = x86.rm.xor %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: xor rax, [rdx+8]
%rm_mov = x86.rm.mov %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: mov rax, [rdx+8]
%rm_cmp = x86.rm.cmp %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.rflags<rflags>
// CHECK-NEXT: cmp rax, [rdx+8]
%rm_lea = x86.rm.lea %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: lea rax, [rdx+8]

%ri_add = x86.ri.add %0, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: add rax, 2
%ri_sub = x86.ri.sub %0, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: sub rax, 2
%ri_and = x86.ri.and %0, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: and rax, 2
%ri_or = x86.ri.or %0, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: or rax, 2
%ri_xor = x86.ri.xor %0, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: xor rax, 2
%ri_mov = x86.ri.mov %0, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: mov rax, 2
%ri_cmp = x86.ri.cmp %0, 2 : (!x86.reg<rax>) -> !x86.rflags<rflags>
// CHECK-NEXT: cmp rax, 2

x86.mr.add %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK-NEXT: add [rax], rdx
x86.mr.add %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK-NEXT: add [rax+8], rdx
x86.mr.sub %0, %1, -8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK-NEXT: sub [rax-8], rdx
x86.mr.and %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK-NEXT: and [rax+8], rdx
x86.mr.or %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK-NEXT: or [rax+8], rdx
x86.mr.xor %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK-NEXT: xor [rax+8], rdx
x86.mr.mov %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK-NEXT: mov [rax+8], rdx
%mr_cmp = x86.mr.cmp %0, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.rflags<rflags>
// CHECK-NEXT: cmp [rax+8], rdx

x86.mi.add %0, 2 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: add [rax], 2
x86.mi.add %0, 2, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: add [rax+8], 2
x86.mi.sub %0, 2, -8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: sub [rax-8], 2
x86.mi.and %0, 2, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: and [rax+8], 2
x86.mi.or %0, 2, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: or [rax+8], 2
x86.mi.xor %0, 2, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: xor [rax+8], 2
x86.mi.mov %0, 2, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: mov [rax+8], 2
%mi_cmp = x86.mi.cmp %0, 2, 8 : (!x86.reg<rax>) -> !x86.rflags<rflags>
// CHECK-NEXT: cmp [rax+8], 2

%rri_imul = x86.rri.imul %1, 2 : (!x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: imul rax, rdx, 2

%rmi_imul_no_offset = x86.rmi.imul %1, 2 : (!x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: imul rax, [rdx], 2
%rmi_imul = x86.rmi.imul %1, 2, 8 : (!x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: imul rax, [rdx+8], 2

%m_push_rsp = x86.m.push %rsp, %0 : (!x86.reg<rsp>, !x86.reg<rax>) -> !x86.reg<rsp>
// CHECK-NEXT: push [rax]
%m_push_rsp2 = x86.m.push %rsp, %0, 8 : (!x86.reg<rsp>, !x86.reg<rax>) -> !x86.reg<rsp>
// CHECK-NEXT: push [rax+8]
%m_pop_rsp = x86.m.pop %rsp, %0, 8 : (!x86.reg<rsp>, !x86.reg<rax>) -> !x86.reg<rsp>
// CHECK-NEXT: pop [rax+8]
x86.m.neg %0 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: neg [rax]
x86.m.neg %0, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: neg [rax+8]
x86.m.not %0, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: not [rax+8]
x86.m.inc %0, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: inc [rax+8]
x86.m.dec %0, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: dec [rax+8]

%m_idiv_rdx, %m_idiv_rax = x86.m.idiv %2, %1, %0 : (!x86.reg<rcx>, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: idiv [rcx]
%m_idiv_rdx2, %m_idiv_rax2 = x86.m.idiv %2, %1, %0, 8 : (!x86.reg<rcx>, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: idiv [rcx+8]
%m_imul_rdx, %m_imul_rax = x86.m.imul %2, %0, 8 : (!x86.reg<rcx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: imul [rcx+8]

x86.directive ".text"
// CHECK-NEXT: .text
x86.directive ".align" "2"
// CHECK-NEXT: .align 2
x86.label "label"
// CHECK-NEXT: label:

func.func @funcyasm() {
    %3 = x86.get_register : () -> !x86.reg<rax>
    %4 = x86.get_register : () -> !x86.reg<rdx>
    %rflags = x86.rr.cmp %3, %4 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.rflags<rflags>
    // CHECK: cmp rax, rdx
    
    x86.s.jmp ^then(%arg : !x86.reg<>)
    // CHECK-NEXT: jmp then
    ^then(%arg : !x86.reg<>):
    x86.label "then"
    // CHECK-NEXT: then:
    x86.s.ja %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else(%arg2 : !x86.reg<>)
    // CHECK-NEXT: ja then
    ^else(%arg2 : !x86.reg<>):
    x86.label "else"
    // CHECK-NEXT: else:
    x86.s.jae %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else2(%arg3 : !x86.reg<>)
    // CHECK-NEXT: jae then
    ^else2(%arg3 : !x86.reg<>):
    x86.label "else2"
    // CHECK-NEXT: else2:
    x86.s.jb %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else3(%arg4 : !x86.reg<>)
    // CHECK-NEXT: jb then
    ^else3(%arg4 : !x86.reg<>):
    x86.label "else3"
    // CHECK-NEXT: else3:
    x86.s.jbe %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else4(%arg5 : !x86.reg<>)
    // CHECK-NEXT: jbe then
    ^else4(%arg5 : !x86.reg<>):
    x86.label "else4"
    // CHECK-NEXT: else4:
    x86.s.jc %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else5(%arg6 : !x86.reg<>)
    // CHECK-NEXT: jc then
    ^else5(%arg6 : !x86.reg<>):
    x86.label "else5"
    // CHECK-NEXT: else5:
    x86.s.je %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else6(%arg7 : !x86.reg<>)
    // CHECK-NEXT: je then
    ^else6(%arg7 : !x86.reg<>):
    x86.label "else6"
    // CHECK-NEXT: else6:
    x86.s.jg %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else7(%arg8 : !x86.reg<>)
    // CHECK-NEXT: jg then
    ^else7(%arg8 : !x86.reg<>):
    x86.label "else7"
    // CHECK-NEXT: else7:
    x86.s.jge %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else8(%arg9 : !x86.reg<>)
    // CHECK-NEXT: jge then
    ^else8(%arg9 : !x86.reg<>):
    x86.label "else8"
    // CHECK-NEXT: else8:
    x86.s.jl %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else9(%arg10 : !x86.reg<>)
    // CHECK-NEXT: jl then
    ^else9(%arg10 : !x86.reg<>):
    x86.label "else9"
    // CHECK-NEXT: else9:
    x86.s.jle %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else10(%arg11 : !x86.reg<>)
    // CHECK-NEXT: jle then
    ^else10(%arg11 : !x86.reg<>):
    x86.label "else10"
    // CHECK-NEXT: else10:
    x86.s.jna %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else11(%arg12 : !x86.reg<>)
    // CHECK-NEXT: jna then
    ^else11(%arg12 : !x86.reg<>):
    x86.label "else11"
    // CHECK-NEXT: else11:
    x86.s.jnae %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else12(%arg13 : !x86.reg<>)
    // CHECK-NEXT: jnae then
    ^else12(%arg13 : !x86.reg<>):
    x86.label "else12"
    // CHECK-NEXT: else12:
    x86.s.jnb %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else13(%arg14 : !x86.reg<>)
    // CHECK-NEXT: jnb then
    ^else13(%arg14 : !x86.reg<>):
    x86.label "else13"
    // CHECK-NEXT: else13:
    x86.s.jnbe %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else14(%arg15 : !x86.reg<>)
    // CHECK-NEXT: jnbe then
    ^else14(%arg15 : !x86.reg<>):
    x86.label "else14"
    // CHECK-NEXT: else14:
    x86.s.jnc %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else15(%arg16 : !x86.reg<>)
    // CHECK-NEXT: jnc then
    ^else15(%arg16 : !x86.reg<>):
    x86.label "else15"
    // CHECK-NEXT: else15:
    x86.s.jne %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else16(%arg17 : !x86.reg<>)
    // CHECK-NEXT: jne then
    ^else16(%arg17 : !x86.reg<>):
    x86.label "else16"
    // CHECK-NEXT: else16:
    x86.s.jng %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else17(%arg18 : !x86.reg<>)
    // CHECK-NEXT: jng then
    ^else17(%arg18 : !x86.reg<>):
    x86.label "else17"
    // CHECK-NEXT: else17:
    x86.s.jnge %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else18(%arg19 : !x86.reg<>)
    // CHECK-NEXT: jnge then
    ^else18(%arg19 : !x86.reg<>):
    x86.label "else18"
    // CHECK-NEXT: else18:
    x86.s.jnl %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else19(%arg20 : !x86.reg<>)
    // CHECK-NEXT: jnl then
    ^else19(%arg20 : !x86.reg<>):
    x86.label "else19"
    // CHECK-NEXT: else19:
    x86.s.jnle %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else20(%arg21 : !x86.reg<>)
    // CHECK-NEXT: jnle then
    ^else20(%arg21 : !x86.reg<>):
    x86.label "else20"
    // CHECK-NEXT: else20:
    x86.s.jno %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else21(%arg22 : !x86.reg<>)
    // CHECK-NEXT: jno then
    ^else21(%arg22 : !x86.reg<>):
    x86.label "else21"
    // CHECK-NEXT: else21:
    x86.s.jnp %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else22(%arg23 : !x86.reg<>)
    // CHECK-NEXT: jnp then
    ^else22(%arg23 : !x86.reg<>):
    x86.label "else22"
    // CHECK-NEXT: else22:
    x86.s.jns %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else23(%arg24 : !x86.reg<>)
    // CHECK-NEXT: jns then
    ^else23(%arg24 : !x86.reg<>):
    x86.label "else23"
    // CHECK-NEXT: else23:
    x86.s.jnz %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else24(%arg25 : !x86.reg<>)
    // CHECK-NEXT: jnz then
    ^else24(%arg25 : !x86.reg<>):
    x86.label "else24"
    // CHECK-NEXT: else24:
    x86.s.jo %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else25(%arg26 : !x86.reg<>)
    // CHECK-NEXT: jo then
    ^else25(%arg26 : !x86.reg<>):
    x86.label "else25"
    // CHECK-NEXT: else25:
    x86.s.jp %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else26(%arg27 : !x86.reg<>)
    // CHECK-NEXT: jp then
    ^else26(%arg27 : !x86.reg<>):
    x86.label "else26"
    // CHECK-NEXT: else26:
    x86.s.jpe %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else27(%arg28 : !x86.reg<>)
    // CHECK-NEXT: jpe then
    ^else27(%arg28 : !x86.reg<>):
    x86.label "else27"
    // CHECK-NEXT: else27:
    x86.s.jpo %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else28(%arg29 : !x86.reg<>)
    // CHECK-NEXT: jpo then
    ^else28(%arg29 : !x86.reg<>):
    x86.label "else28"
    // CHECK-NEXT: else28:
    x86.s.js %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else29(%arg30 : !x86.reg<>)
    // CHECK-NEXT: js then
    ^else29(%arg30 : !x86.reg<>):
    x86.label "else29"
    // CHECK-NEXT: else29:
    x86.s.jz %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg<>), ^else30(%arg31 : !x86.reg<>)
    // CHECK-NEXT: jz then
    ^else30(%arg31 : !x86.reg<>):
    x86.label "else30"
    // CHECK-NEXT: else30:

    x86.s.jmp ^then(%arg : !x86.reg<>)
    // CHECK-NEXT: jmp then
}

%zmm0 = x86.get_avx_register : () -> !x86.avxreg<zmm0>
%zmm1 = x86.get_avx_register : () -> !x86.avxreg<zmm1>
%zmm2 = x86.get_avx_register : () -> !x86.avxreg<zmm2>

%rrr_vfmadd231pd = x86.rrr.vfmadd231pd %zmm0, %zmm1, %zmm2 : (!x86.avxreg<zmm0>, !x86.avxreg<zmm1>, !x86.avxreg<zmm2>) -> !x86.avxreg<zmm0>
// CHECK: vfmadd231pd zmm0, zmm1, zmm2
%rr_vmovapd = x86.rr.vmovapd %zmm0, %zmm1 : (!x86.avxreg<zmm0>, !x86.avxreg<zmm1>) -> !x86.avxreg<zmm0>
// CHECK-NEXT: vmovapd zmm0, zmm1
x86.mr.vmovapd %0, %zmm1, 8 : (!x86.reg<rax>, !x86.avxreg<zmm1>) -> ()
// CHECK-NEXT: vmovapd [rax+8], zmm1
%rm_vbroadcastsd = x86.rm.vbroadcastsd %zmm0, %1, 8 : (!x86.avxreg<zmm0>, !x86.reg<rdx>) -> !x86.avxreg<zmm0>
// CHECK-NEXT: vbroadcastsd zmm0, [rdx+8]
