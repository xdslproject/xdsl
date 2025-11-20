// RUN: xdsl-opt -t x86-asm %s | filecheck %s

// CHECK-NEXT: .intel_syntax noprefix

%0 = x86.get_register : () -> !x86.reg<rax>
%1 = x86.get_register : () -> !x86.reg<rdx>
%2 = x86.get_register : () -> !x86.reg<rcx>
%rsp = x86.get_register : () -> !x86.reg<rsp>
%rax = x86.get_register : () -> !x86.reg<rax>

%rs_add = x86.rs.add %0, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: add rax, rdx
%rs_fadd = x86.rs.fadd %rs_add, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: fadd rax, rdx
%rr_sub = x86.rs.sub %rs_fadd, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: sub rax, rdx
%rs_imul = x86.rs.imul %rr_sub, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: imul rax, rdx
%rs_fmul = x86.rs.fmul %rs_imul, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: fmul rax, rdx
%rr_and = x86.rs.and %rs_fmul, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: and rax, rdx
%rr_or = x86.rs.or %rr_and, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: or rax, rdx
%rr_xor = x86.rs.xor %rr_or, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: xor rax, rdx
%ds_mov = x86.ds.mov %1 : (!x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: mov rax, rdx
%rr_cmp = x86.ss.cmp %rax, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.rflags<rflags>
// CHECK-NEXT: cmp rax, rdx
%r_pushrsp = x86.s.push %rsp, %rr_xor : (!x86.reg<rsp>, !x86.reg<rax>) -> !x86.reg<rsp>
// CHECK-NEXT: push rax
%r_poprsp, %r_pop = x86.d.pop %rsp : (!x86.reg<rsp>) -> (!x86.reg<rsp>, !x86.reg<rax>)
// CHECK-NEXT: pop rax
%r_neg = x86.r.neg %r_pop : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: neg rax
%r_not = x86.r.not %r_neg : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: not rax
%r_inc = x86.r.inc %r_not : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: inc rax
%r_dec = x86.r.dec %r_inc : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: dec rax

%r_idiv_rdx, %r_idiv_rax = x86.s.idiv %2, %1, %r_dec : (!x86.reg<rcx>, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: idiv rcx
%r_imul_rdx, %r_imul_rax = x86.s.imul %2, %r_idiv_rax : (!x86.reg<rcx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: imul rcx

%rm_add_no_offset = x86.rm.add %r_imul_rax, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: add rax, [rdx]
%rm_add_offset_zero = x86.rm.add %rm_add_no_offset, %1, 0 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: add rax, [rdx]
%rm_add = x86.rm.add %rm_add_offset_zero, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: add rax, [rdx+8]
%rm_sub = x86.rm.sub %rm_add, %1, -8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: sub rax, [rdx-8]
%rm_imul = x86.rm.imul %rm_sub, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: imul rax, [rdx+8]
%rm_and = x86.rm.and %rm_imul, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: and rax, [rdx+8]
%rm_or = x86.rm.or %rm_and, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: or rax, [rdx+8]
%rm_xor = x86.rm.xor %rm_or, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: xor rax, [rdx+8]
%rm_mov = x86.dm.mov %1, 8 : (!x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: mov rax, [rdx+8]
%rm_cmp = x86.sm.cmp %rm_mov, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.rflags<rflags>
// CHECK-NEXT: cmp rax, [rdx+8]
%rm_lea = x86.dm.lea %1, 8 : (!x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: lea rax, [rdx+8]

%ri_add = x86.ri.add %rm_lea, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: add rax, 2
%ri_sub = x86.ri.sub %ri_add, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: sub rax, 2
%ri_and = x86.ri.and %ri_sub, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: and rax, 2
%ri_or = x86.ri.or %ri_and, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: or rax, 2
%ri_xor = x86.ri.xor %ri_or, 2 : (!x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT: xor rax, 2
%di_mov = x86.di.mov 2 : () -> !x86.reg<rax>
// CHECK-NEXT: mov rax, 2
%ri_cmp = x86.si.cmp %di_mov, 2 : (!x86.reg<rax>) -> !x86.rflags<rflags>
// CHECK-NEXT: cmp rax, 2

x86.ms.add %rax, %1 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK-NEXT: add [rax], rdx
x86.ms.add %rax, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK-NEXT: add [rax+8], rdx
x86.ms.sub %rax, %1, -8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK-NEXT: sub [rax-8], rdx
x86.ms.and %rax, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK-NEXT: and [rax+8], rdx
x86.ms.or %rax, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK-NEXT: or [rax+8], rdx
x86.ms.xor %rax, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK-NEXT: xor [rax+8], rdx
x86.ms.mov %rax, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> ()
// CHECK-NEXT: mov [rax+8], rdx
%mr_cmp = x86.ms.cmp %rax, %1, 8 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.rflags<rflags>
// CHECK-NEXT: cmp [rax+8], rdx

x86.mi.add %rax, 2, 0 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: add [rax], 2
x86.mi.add %rax, 2 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: add [rax], 2
x86.mi.add %rax, 2, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: add [rax+8], 2
x86.mi.sub %rax, 2, -8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: sub [rax-8], 2
x86.mi.and %rax, 2, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: and [rax+8], 2
x86.mi.or %rax, 2, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: or [rax+8], 2
x86.mi.xor %rax, 2, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: xor [rax+8], 2
x86.mi.mov %rax, 2, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: mov [rax+8], 2
%mi_cmp = x86.mi.cmp %rax, 2, 8 : (!x86.reg<rax>) -> !x86.rflags<rflags>
// CHECK-NEXT: cmp [rax+8], 2

%rri_imul = x86.dsi.imul %1, 2 : (!x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: imul rax, rdx, 2

%rmi_imul_no_offset = x86.dmi.imul %1, 2 : (!x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: imul rax, [rdx], 2
%rmi_imul_offset_zero = x86.dmi.imul %1, 2, 0 : (!x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: imul rax, [rdx], 2
%rmi_imul = x86.dmi.imul %1, 2, 8 : (!x86.reg<rdx>) -> !x86.reg<rax>
// CHECK-NEXT: imul rax, [rdx+8], 2

%m_push_rsp = x86.m.push %rsp, %rax : (!x86.reg<rsp>, !x86.reg<rax>) -> !x86.reg<rsp>
// CHECK-NEXT: push [rax]
%m_push_rsp0 = x86.m.push %rsp, %rax, 0 : (!x86.reg<rsp>, !x86.reg<rax>) -> !x86.reg<rsp>
// CHECK-NEXT: push [rax]
%m_push_rsp2 = x86.m.push %rsp, %rax, 8 : (!x86.reg<rsp>, !x86.reg<rax>) -> !x86.reg<rsp>
// CHECK-NEXT: push [rax+8]
%m_pop_rsp = x86.m.pop %rsp, %rax, 8 : (!x86.reg<rsp>, !x86.reg<rax>) -> !x86.reg<rsp>
// CHECK-NEXT: pop [rax+8]
x86.m.neg %rax : (!x86.reg<rax>) -> ()
// CHECK-NEXT: neg [rax]
x86.m.neg %rax, 0 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: neg [rax]
x86.m.neg %rax, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: neg [rax+8]
x86.m.not %rax, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: not [rax+8]
x86.m.inc %rax, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: inc [rax+8]
x86.m.dec %rax, 8 : (!x86.reg<rax>) -> ()
// CHECK-NEXT: dec [rax+8]

%m_idiv_rdx, %m_idiv_rax = x86.m.idiv %2, %1, %rmi_imul : (!x86.reg<rcx>, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: idiv [rcx]
%m_idiv_rdx0, %m_idiv_rax0 = x86.m.idiv %2, %1, %m_idiv_rax, 0 : (!x86.reg<rcx>, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: idiv [rcx]
%m_idiv_rdx2, %m_idiv_rax2 = x86.m.idiv %2, %1, %m_idiv_rax0, 8 : (!x86.reg<rcx>, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: idiv [rcx+8]
%m_imul_rdx, %m_imul_rax = x86.m.imul %2, %m_idiv_rax2, 8 : (!x86.reg<rcx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: imul [rcx+8]

x86.directive ".text"
// CHECK-NEXT: .text
x86.directive ".align" "2"
// CHECK-NEXT: .align 2
x86.label "label"
// CHECK-NEXT: label:

x86_func.func @funcyasm() {
    %3 = x86.get_register : () -> !x86.reg<rax>
    %4 = x86.get_register : () -> !x86.reg<rdx>
    %rflags = x86.ss.cmp %3, %4 : (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.rflags<rflags>
    // CHECK: cmp rax, rdx

    x86.fallthrough ^fallthrough()
    // CHECK-NOT: jmp
    ^fallthrough:
    x86.label "fallthrough"
    // CHECK-NEXT: fallthrough:
    x86.c.jmp ^then(%arg : !x86.reg)
    // CHECK-NEXT: jmp then
    ^then(%arg : !x86.reg):
    x86.label "then"
    // CHECK-NEXT: then:
    x86.c.ja %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else(%arg2 : !x86.reg)
    // CHECK-NEXT: ja then
    ^else(%arg2 : !x86.reg):
    x86.label "else"
    // CHECK-NEXT: else:
    x86.c.jae %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else2(%arg3 : !x86.reg)
    // CHECK-NEXT: jae then
    ^else2(%arg3 : !x86.reg):
    x86.label "else2"
    // CHECK-NEXT: else2:
    x86.c.jb %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else3(%arg4 : !x86.reg)
    // CHECK-NEXT: jb then
    ^else3(%arg4 : !x86.reg):
    x86.label "else3"
    // CHECK-NEXT: else3:
    x86.c.jbe %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else4(%arg5 : !x86.reg)
    // CHECK-NEXT: jbe then
    ^else4(%arg5 : !x86.reg):
    x86.label "else4"
    // CHECK-NEXT: else4:
    x86.c.jc %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else5(%arg6 : !x86.reg)
    // CHECK-NEXT: jc then
    ^else5(%arg6 : !x86.reg):
    x86.label "else5"
    // CHECK-NEXT: else5:
    x86.c.je %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else6(%arg7 : !x86.reg)
    // CHECK-NEXT: je then
    ^else6(%arg7 : !x86.reg):
    x86.label "else6"
    // CHECK-NEXT: else6:
    x86.c.jg %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else7(%arg8 : !x86.reg)
    // CHECK-NEXT: jg then
    ^else7(%arg8 : !x86.reg):
    x86.label "else7"
    // CHECK-NEXT: else7:
    x86.c.jge %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else8(%arg9 : !x86.reg)
    // CHECK-NEXT: jge then
    ^else8(%arg9 : !x86.reg):
    x86.label "else8"
    // CHECK-NEXT: else8:
    x86.c.jl %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else9(%arg10 : !x86.reg)
    // CHECK-NEXT: jl then
    ^else9(%arg10 : !x86.reg):
    x86.label "else9"
    // CHECK-NEXT: else9:
    x86.c.jle %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else10(%arg11 : !x86.reg)
    // CHECK-NEXT: jle then
    ^else10(%arg11 : !x86.reg):
    x86.label "else10"
    // CHECK-NEXT: else10:
    x86.c.jna %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else11(%arg12 : !x86.reg)
    // CHECK-NEXT: jna then
    ^else11(%arg12 : !x86.reg):
    x86.label "else11"
    // CHECK-NEXT: else11:
    x86.c.jnae %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else12(%arg13 : !x86.reg)
    // CHECK-NEXT: jnae then
    ^else12(%arg13 : !x86.reg):
    x86.label "else12"
    // CHECK-NEXT: else12:
    x86.c.jnb %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else13(%arg14 : !x86.reg)
    // CHECK-NEXT: jnb then
    ^else13(%arg14 : !x86.reg):
    x86.label "else13"
    // CHECK-NEXT: else13:
    x86.c.jnbe %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else14(%arg15 : !x86.reg)
    // CHECK-NEXT: jnbe then
    ^else14(%arg15 : !x86.reg):
    x86.label "else14"
    // CHECK-NEXT: else14:
    x86.c.jnc %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else15(%arg16 : !x86.reg)
    // CHECK-NEXT: jnc then
    ^else15(%arg16 : !x86.reg):
    x86.label "else15"
    // CHECK-NEXT: else15:
    x86.c.jne %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else16(%arg17 : !x86.reg)
    // CHECK-NEXT: jne then
    ^else16(%arg17 : !x86.reg):
    x86.label "else16"
    // CHECK-NEXT: else16:
    x86.c.jng %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else17(%arg18 : !x86.reg)
    // CHECK-NEXT: jng then
    ^else17(%arg18 : !x86.reg):
    x86.label "else17"
    // CHECK-NEXT: else17:
    x86.c.jnge %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else18(%arg19 : !x86.reg)
    // CHECK-NEXT: jnge then
    ^else18(%arg19 : !x86.reg):
    x86.label "else18"
    // CHECK-NEXT: else18:
    x86.c.jnl %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else19(%arg20 : !x86.reg)
    // CHECK-NEXT: jnl then
    ^else19(%arg20 : !x86.reg):
    x86.label "else19"
    // CHECK-NEXT: else19:
    x86.c.jnle %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else20(%arg21 : !x86.reg)
    // CHECK-NEXT: jnle then
    ^else20(%arg21 : !x86.reg):
    x86.label "else20"
    // CHECK-NEXT: else20:
    x86.c.jno %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else21(%arg22 : !x86.reg)
    // CHECK-NEXT: jno then
    ^else21(%arg22 : !x86.reg):
    x86.label "else21"
    // CHECK-NEXT: else21:
    x86.c.jnp %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else22(%arg23 : !x86.reg)
    // CHECK-NEXT: jnp then
    ^else22(%arg23 : !x86.reg):
    x86.label "else22"
    // CHECK-NEXT: else22:
    x86.c.jns %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else23(%arg24 : !x86.reg)
    // CHECK-NEXT: jns then
    ^else23(%arg24 : !x86.reg):
    x86.label "else23"
    // CHECK-NEXT: else23:
    x86.c.jnz %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else24(%arg25 : !x86.reg)
    // CHECK-NEXT: jnz then
    ^else24(%arg25 : !x86.reg):
    x86.label "else24"
    // CHECK-NEXT: else24:
    x86.c.jo %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else25(%arg26 : !x86.reg)
    // CHECK-NEXT: jo then
    ^else25(%arg26 : !x86.reg):
    x86.label "else25"
    // CHECK-NEXT: else25:
    x86.c.jp %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else26(%arg27 : !x86.reg)
    // CHECK-NEXT: jp then
    ^else26(%arg27 : !x86.reg):
    x86.label "else26"
    // CHECK-NEXT: else26:
    x86.c.jpe %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else27(%arg28 : !x86.reg)
    // CHECK-NEXT: jpe then
    ^else27(%arg28 : !x86.reg):
    x86.label "else27"
    // CHECK-NEXT: else27:
    x86.c.jpo %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else28(%arg29 : !x86.reg)
    // CHECK-NEXT: jpo then
    ^else28(%arg29 : !x86.reg):
    x86.label "else28"
    // CHECK-NEXT: else28:
    x86.c.js %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else29(%arg30 : !x86.reg)
    // CHECK-NEXT: js then
    ^else29(%arg30 : !x86.reg):
    x86.label "else29"
    // CHECK-NEXT: else29:
    x86.c.jz %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else30(%arg31 : !x86.reg)
    // CHECK-NEXT: jz then
    ^else30(%arg31 : !x86.reg):
    x86.label "else30"
    // CHECK-NEXT: else30:

    x86.c.jmp ^then(%arg : !x86.reg)
    // CHECK-NEXT: jmp then
}

%xmm0 = x86.get_avx_register : () -> !x86.ssereg<xmm0>
%xmm1 = x86.get_avx_register : () -> !x86.ssereg<xmm1>
%xmm2 = x86.get_avx_register : () -> !x86.ssereg<xmm2>

%rrr_vfmadd231pd_sse = x86.rss.vfmadd231pd %xmm0, %xmm1, %xmm2 : (!x86.ssereg<xmm0>, !x86.ssereg<xmm1>, !x86.ssereg<xmm2>) -> !x86.ssereg<xmm0>
// CHECK: vfmadd231pd xmm0, xmm1, xmm2
%rrm_vfmadd231pd_sse = x86.rsm.vfmadd231pd %rrr_vfmadd231pd_sse, %xmm1, %1, 8 : (!x86.ssereg<xmm0>, !x86.ssereg<xmm1>, !x86.reg<rdx>) -> !x86.ssereg<xmm0>
// CHECK: vfmadd231pd xmm0, xmm1, [rdx+8]
%rrm_vfmadd231pd_sse_no_offset = x86.rsm.vfmadd231pd %rrm_vfmadd231pd_sse, %xmm1, %1 : (!x86.ssereg<xmm0>, !x86.ssereg<xmm1>, !x86.reg<rdx>) -> !x86.ssereg<xmm0>
// CHECK: vfmadd231pd xmm0, xmm1, [rdx]
%rrm_vfmadd231ps_sse = x86.rsm.vfmadd231ps %rrm_vfmadd231pd_sse_no_offset, %xmm1, %1, 1 : (!x86.ssereg<xmm0>, !x86.ssereg<xmm1>, !x86.reg<rdx>) -> !x86.ssereg<xmm0>
// CHECK: vfmadd231ps xmm0, xmm1, [rdx+1]
%dss_addpd_sse = x86.dss.addpd %xmm1, %xmm2 : (!x86.ssereg<xmm1>, !x86.ssereg<xmm2>) -> !x86.ssereg<xmm0>
// CHECK-NEXT: addpd xmm0, xmm1, xmm2
%dss_addps_sse = x86.dss.addps %xmm1, %xmm2 : (!x86.ssereg<xmm1>, !x86.ssereg<xmm2>) -> !x86.ssereg<xmm0>
// CHECK-NEXT: addps xmm0, xmm1, xmm2
%rm_vbroadcastsd_sse0 = x86.dm.vbroadcastsd %1, 0 : (!x86.reg<rdx>) -> !x86.ssereg<xmm0>
// CHECK-NEXT: vbroadcastsd xmm0, [rdx]
%rm_vbroadcastsd_sse = x86.dm.vbroadcastsd %1, 8 : (!x86.reg<rdx>) -> !x86.ssereg<xmm0>
// CHECK-NEXT: vbroadcastsd xmm0, [rdx+8]

%ymm0 = x86.get_avx_register : () -> !x86.avx2reg<ymm0>
%ymm1 = x86.get_avx_register : () -> !x86.avx2reg<ymm1>
%ymm2 = x86.get_avx_register : () -> !x86.avx2reg<ymm2>

%rrr_vfmadd231pd_avx2 = x86.rss.vfmadd231pd %ymm0, %ymm1, %ymm2 : (!x86.avx2reg<ymm0>, !x86.avx2reg<ymm1>, !x86.avx2reg<ymm2>) -> !x86.avx2reg<ymm0>
// CHECK: vfmadd231pd ymm0, ymm1, ymm2
%rrm_vfmadd231pd_avx2 = x86.rsm.vfmadd231pd %rrr_vfmadd231pd_avx2, %ymm1, %1, 8 : (!x86.avx2reg<ymm0>, !x86.avx2reg<ymm1>, !x86.reg<rdx>) -> !x86.avx2reg<ymm0>
// CHECK: vfmadd231pd ymm0, ymm1, [rdx+8]
%rrm_vfmadd231pd_avx2_no_offset = x86.rsm.vfmadd231pd %rrm_vfmadd231pd_avx2, %ymm1, %1 : (!x86.avx2reg<ymm0>, !x86.avx2reg<ymm1>, !x86.reg<rdx>) -> !x86.avx2reg<ymm0>
// CHECK: vfmadd231pd ymm0, ymm1, [rdx]
%rrm_vfmadd231ps_avx2 = x86.rsm.vfmadd231ps %rrm_vfmadd231pd_avx2_no_offset, %ymm1, %1, 1 : (!x86.avx2reg<ymm0>, !x86.avx2reg<ymm1>, !x86.reg<rdx>) -> !x86.avx2reg<ymm0>
// CHECK: vfmadd231ps ymm0, ymm1, [rdx+1]
%dss_addpd_avx2 = x86.dss.addpd %ymm1, %ymm2 : (!x86.avx2reg<ymm1>, !x86.avx2reg<ymm2>) -> !x86.avx2reg<ymm0>
// CHECK-NEXT: addpd ymm0, ymm1, ymm2
%dss_addps_avx2 = x86.dss.addps %ymm1, %ymm2 : (!x86.avx2reg<ymm1>, !x86.avx2reg<ymm2>) -> !x86.avx2reg<ymm0>
// CHECK-NEXT: addps ymm0, ymm1, ymm2
%rm_vbroadcastsd_avx20 = x86.dm.vbroadcastsd %1, 0 : (!x86.reg<rdx>) -> !x86.avx2reg<ymm0>
// CHECK-NEXT: vbroadcastsd ymm0, [rdx]
%rm_vbroadcastsd_avx2 = x86.dm.vbroadcastsd %1, 8 : (!x86.reg<rdx>) -> !x86.avx2reg<ymm0>
// CHECK-NEXT: vbroadcastsd ymm0, [rdx+8]
%ds_vpbroadcastd_avx2 = x86.ds.vpbroadcastd %rax : (!x86.reg<rax>) -> !x86.avx2reg<ymm0>
// CHECK-NEXT: vpbroadcastd ymm0, rax
%ds_vpbroadcastq_avx2 = x86.ds.vpbroadcastq %rax : (!x86.reg<rax>) -> !x86.avx2reg<ymm0>
// CHECK-NEXT: vpbroadcastq ymm0, rax

%zmm0 = x86.get_avx_register : () -> !x86.avx512reg<zmm0>
%zmm1 = x86.get_avx_register : () -> !x86.avx512reg<zmm1>
%zmm2 = x86.get_avx_register : () -> !x86.avx512reg<zmm2>
%k1 = x86.get_mask_register: () -> !x86.avx512maskreg<k1>

%rrr_vfmadd231pd_avx512 = x86.rss.vfmadd231pd %zmm0, %zmm1, %zmm2 : (!x86.avx512reg<zmm0>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm2>) -> !x86.avx512reg<zmm0>
// CHECK: vfmadd231pd zmm0, zmm1, zmm2
%rrrk_vfmadd231pd_avx512_no_z = x86.rssk.vfmadd231pd %rrr_vfmadd231pd_avx512, %zmm1, %zmm2, %k1 : (!x86.avx512reg<zmm0>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm2>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK: vfmadd231pd zmm0 {k1}, zmm1, zmm2
%rrrk_vfmadd231pd_avx512_z = x86.rssk.vfmadd231pd %rrrk_vfmadd231pd_avx512_no_z, %zmm1, %zmm2, %k1 {z} : (!x86.avx512reg<zmm0>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm2>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK: vfmadd231pd zmm0 {k1}{z}, zmm1, zmm2
%rrm_vfmadd231pd_avx512 = x86.rsm.vfmadd231pd %rrrk_vfmadd231pd_avx512_z, %zmm1, %1, 8 : (!x86.avx512reg<zmm0>, !x86.avx512reg<zmm1>, !x86.reg<rdx>) -> !x86.avx512reg<zmm0>
// CHECK: vfmadd231pd zmm0, zmm1, [rdx+8]
%rrm_vfmadd231pd_avx512_no_offset = x86.rsm.vfmadd231pd %rrm_vfmadd231pd_avx512, %zmm1, %1 : (!x86.avx512reg<zmm0>, !x86.avx512reg<zmm1>, !x86.reg<rdx>) -> !x86.avx512reg<zmm0>
// CHECK: vfmadd231pd zmm0, zmm1, [rdx]
%rrm_vfmadd231ps_avx512 = x86.rsm.vfmadd231ps %rrm_vfmadd231pd_avx512_no_offset, %zmm1, %1, 1 : (!x86.avx512reg<zmm0>, !x86.avx512reg<zmm1>, !x86.reg<rdx>) -> !x86.avx512reg<zmm0>
// CHECK: vfmadd231ps zmm0, zmm1, [rdx+1]
%dss_addpd_avx512 = x86.dss.addpd %zmm1, %zmm2 : (!x86.avx512reg<zmm1>, !x86.avx512reg<zmm2>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT: addpd zmm0, zmm1, zmm2
%dss_addps_avx512 = x86.dss.addps %zmm1, %zmm2 : (!x86.avx512reg<zmm1>, !x86.avx512reg<zmm2>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT: addps zmm0, zmm1, zmm2
%rm_vbroadcastsd_avx5120 = x86.dm.vbroadcastsd %1, 0 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT: vbroadcastsd zmm0, [rdx]
%rm_vbroadcastsd_avx512 = x86.dm.vbroadcastsd %1, 8 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT: vbroadcastsd zmm0, [rdx+8]

%rm_vbroadcastss_avx512 = x86.dm.vbroadcastss %1 : (!x86.reg<rdx>) -> (!x86.avx512reg<zmm0>)
// CHECK: vbroadcastss zmm0, [rdx]
%rm_vbroadcastss_avx2 = x86.dm.vbroadcastss %1 : (!x86.reg<rdx>) -> (!x86.avx2reg<ymm0>)
// CHECK-NEXT: vbroadcastss ymm0, [rdx]
%rm_vbroadcastss_sse = x86.dm.vbroadcastss %1 : (!x86.reg<rdx>) -> (!x86.ssereg<xmm0>)
// CHECK-NEXT: vbroadcastss xmm0, [rdx]

// ---- vmovapd ----
%dm_vmovapd_sse = x86.dm.vmovapd %1 : (!x86.reg<rdx>) -> !x86.ssereg<xmm0>
// CHECK: vmovapd xmm0, [rdx]
%dm_vmovapd_avx2 = x86.dm.vmovapd %1 : (!x86.reg<rdx>) -> !x86.avx2reg<ymm0>
// CHECK: vmovapd ymm0, [rdx]
%dm_vmovapd_avx512 = x86.dm.vmovapd %1 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm0>
// CHECK: vmovapd zmm0, [rdx]

%ds_vmovapd_sse = x86.ds.vmovapd %xmm1 : (!x86.ssereg<xmm1>) -> !x86.ssereg<xmm0>
// CHECK: vmovapd xmm0, xmm1
%ds_vmovapd_avx2 = x86.ds.vmovapd %ymm1 : (!x86.avx2reg<ymm1>) -> !x86.avx2reg<ymm0>
// CHECK: vmovapd ymm0, ymm1
%ds_vmovapd_avx512 = x86.ds.vmovapd %zmm1 : (!x86.avx512reg<zmm1>) -> !x86.avx512reg<zmm0>
// CHECK: vmovapd zmm0, zmm1
%ds_vmovapd_avx512_mask = x86.dsk.vmovapd %zmm1, %k1 : (!x86.avx512reg<zmm1>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK: vmovapd zmm0 {k1}, zmm1
%ds_vmovapd_avx512_mask_z = x86.dsk.vmovapd %zmm1, %k1 {z} : (!x86.avx512reg<zmm1>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT: vmovapd zmm0 {k1}{z}, zmm1

x86.ms.vmovapd %1, %xmm1, 8 : (!x86.reg<rdx>, !x86.ssereg<xmm1>) -> ()
// CHECK: vmovapd [rdx+8], xmm1
x86.ms.vmovapd %1, %ymm1, 8 : (!x86.reg<rdx>, !x86.avx2reg<ymm1>) -> ()
// CHECK: vmovapd [rdx+8], ymm1
x86.ms.vmovapd %1, %zmm1, 8 : (!x86.reg<rdx>, !x86.avx512reg<zmm1>) -> ()
// CHECK: vmovapd [rdx+8], zmm1


// ---- vmovaps ----
%dm_vmovaps_sse = x86.dm.vmovaps %1 : (!x86.reg<rdx>) -> !x86.ssereg<xmm0>
// CHECK: vmovaps xmm0, [rdx]
%dm_vmovaps_avx2 = x86.dm.vmovaps %1 : (!x86.reg<rdx>) -> !x86.avx2reg<ymm0>
// CHECK: vmovaps ymm0, [rdx]
%dm_vmovaps_avx512 = x86.dm.vmovaps %1 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm0>
// CHECK: vmovaps zmm0, [rdx]

%ds_vmovaps_sse = x86.ds.vmovaps %xmm1 : (!x86.ssereg<xmm1>) -> !x86.ssereg<xmm0>
// CHECK: vmovaps xmm0, xmm1
%ds_vmovaps_avx2 = x86.ds.vmovaps %ymm1 : (!x86.avx2reg<ymm1>) -> !x86.avx2reg<ymm0>
// CHECK: vmovaps ymm0, ymm1
%ds_vmovaps_avx512 = x86.ds.vmovaps %zmm1 : (!x86.avx512reg<zmm1>) -> !x86.avx512reg<zmm0>
// CHECK: vmovaps zmm0, zmm1

x86.ms.vmovaps %1, %xmm1, 8 : (!x86.reg<rdx>, !x86.ssereg<xmm1>) -> ()
// CHECK: vmovaps [rdx+8], xmm1
x86.ms.vmovaps %1, %ymm1, 8 : (!x86.reg<rdx>, !x86.avx2reg<ymm1>) -> ()
// CHECK: vmovaps [rdx+8], ymm1
x86.ms.vmovaps %1, %zmm1, 8 : (!x86.reg<rdx>, !x86.avx512reg<zmm1>) -> ()
// CHECK: vmovaps [rdx+8], zmm1


// ---- vmovupd ----
%dm_vmovupd_sse = x86.dm.vmovupd %1 : (!x86.reg<rdx>) -> !x86.ssereg<xmm0>
// CHECK: vmovupd xmm0, [rdx]
%dm_vmovupd_avx2 = x86.dm.vmovupd %1 : (!x86.reg<rdx>) -> !x86.avx2reg<ymm0>
// CHECK: vmovupd ymm0, [rdx]
%dm_vmovupd_avx512 = x86.dm.vmovupd %1 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm0>
// CHECK: vmovupd zmm0, [rdx]

x86.ms.vmovupd %1, %xmm1, 8 : (!x86.reg<rdx>, !x86.ssereg<xmm1>) -> ()
// CHECK: vmovupd [rdx+8], xmm1
x86.ms.vmovupd %1, %ymm1, 8 : (!x86.reg<rdx>, !x86.avx2reg<ymm1>) -> ()
// CHECK: vmovupd [rdx+8], ymm1
x86.ms.vmovupd %1, %zmm1, 8 : (!x86.reg<rdx>, !x86.avx512reg<zmm1>) -> ()
// CHECK: vmovupd [rdx+8], zmm1


// ---- vmovups ----
%dm_vmovups_sse = x86.dm.vmovups %1 : (!x86.reg<rdx>) -> !x86.ssereg<xmm0>
// CHECK: vmovups xmm0, [rdx]
%dm_vmovups_avx2 = x86.dm.vmovups %1 : (!x86.reg<rdx>) -> !x86.avx2reg<ymm0>
// CHECK: vmovups ymm0, [rdx]
%dm_vmovups_avx512 = x86.dm.vmovups %1 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm0>
// CHECK: vmovups zmm0, [rdx]

x86.ms.vmovups %1, %xmm1, 8 : (!x86.reg<rdx>, !x86.ssereg<xmm1>) -> ()
// CHECK: vmovups [rdx+8], xmm1
x86.ms.vmovups %1, %ymm1, 8 : (!x86.reg<rdx>, !x86.avx2reg<ymm1>) -> ()
// CHECK: vmovups [rdx+8], ymm1
x86.ms.vmovups %1, %zmm1, 8 : (!x86.reg<rdx>, !x86.avx512reg<zmm1>) -> ()
// CHECK: vmovups [rdx+8], zmm1

// ---- vmovntpd ----
x86.ms.vmovntpd %1, %xmm1, 8 : (!x86.reg<rdx>, !x86.ssereg<xmm1>) -> ()
// CHECK: vmovntpd [rdx+8], xmm1
x86.ms.vmovntpd %1, %ymm1, 8 : (!x86.reg<rdx>, !x86.avx2reg<ymm1>) -> ()
// CHECK: vmovntpd [rdx+8], ymm1
x86.ms.vmovntpd %1, %zmm1, 8 : (!x86.reg<rdx>, !x86.avx512reg<zmm1>) -> ()
// CHECK: vmovntpd [rdx+8], zmm1

// ---- vmovntps ----
x86.ms.vmovntps %1, %xmm1, 8 : (!x86.reg<rdx>, !x86.ssereg<xmm1>) -> ()
// CHECK: vmovntps [rdx+8], xmm1
x86.ms.vmovntps %1, %ymm1, 8 : (!x86.reg<rdx>, !x86.avx2reg<ymm1>) -> ()
// CHECK: vmovntps [rdx+8], ymm1
x86.ms.vmovntps %1, %zmm1, 8 : (!x86.reg<rdx>, !x86.avx512reg<zmm1>) -> ()
// CHECK: vmovntps [rdx+8], zmm1

%rrr_vfmadd231ps_sse = x86.rss.vfmadd231ps %ds_vmovapd_sse, %xmm1, %xmm2 : (!x86.ssereg<xmm0>, !x86.ssereg<xmm1>, !x86.ssereg<xmm2>) -> !x86.ssereg<xmm0>
// CHECK: vfmadd231ps xmm0, xmm1, xmm2
%rrr_vfmadd231ps_avx2 = x86.rss.vfmadd231ps %ds_vmovapd_avx2, %ymm1, %ymm2 : (!x86.avx2reg<ymm0>, !x86.avx2reg<ymm1>, !x86.avx2reg<ymm2>) -> !x86.avx2reg<ymm0>
// CHECK-NEXT: vfmadd231ps ymm0, ymm1, ymm2
%rrr_vfmadd231ps_avx512 = x86.rss.vfmadd231ps %ds_vmovapd_avx512, %zmm1, %zmm2 : (!x86.avx512reg<zmm0>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm2>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT: vfmadd231ps zmm0, zmm1, zmm2

%shuf_res = x86.dssi.shufps %zmm1, %zmm2, 170 : (!x86.avx512reg<zmm1>, !x86.avx512reg<zmm2>) -> !x86.avx512reg<zmm0>
// CHECK: shufps zmm0, zmm1, zmm2, 170
