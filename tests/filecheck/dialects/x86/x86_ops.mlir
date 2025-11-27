// RUN: XDSL_ROUNDTRIP
%a = "test.op"() : () -> !x86.reg<rax>
// CHECK: %{{.*}} = "test.op"() : () -> !x86.reg<rax>

%0, %1, %2 = "test.op"() : () -> (!x86.reg, !x86.reg, !x86.reg)
%rsp = "test.op"() : () -> !x86.reg<rsp>
%rax = "test.op"() : () -> !x86.reg<rax>
%rdx = "test.op"() : () -> !x86.reg<rdx>

%rs_add = x86.rs.add %0, %1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK: %{{.*}} = x86.rs.add %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%rs_fadd = x86.rs.fadd %rs_add, %1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK: %{{.*}} = x86.rs.fadd %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%rr_sub = x86.rs.sub %rs_fadd, %1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rs.sub %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%rr_mul = x86.rs.imul %rr_sub, %1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rs.imul %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%rr_fmul = x86.rs.fmul %rr_mul, %1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rs.fmul %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%rr_and = x86.rs.and %rr_fmul, %1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rs.and %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%rr_or = x86.rs.or %rr_and, %1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rs.or %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%rr_xor = x86.rs.xor %rr_or, %1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rs.xor %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%ds_mov = x86.ds.mov %1 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %ds_mov = x86.ds.mov %{{.*}} : (!x86.reg) -> !x86.reg
%rr_cmp = x86.ss.cmp %rr_xor, %1 : (!x86.reg, !x86.reg) -> !x86.rflags<rflags>
// CHECK: %{{.*}} = x86.ss.cmp %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.rflags

%r_pushrsp = x86.s.push %rsp, %1 : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
// CHECK-NEXT: %{{.*}} = x86.s.push %rsp, %{{.*}} : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
%r_poprsp, %r_pop = x86.d.pop %rsp : (!x86.reg<rsp>) -> (!x86.reg<rsp>, !x86.reg)
// CHECK-NEXT: %{{.*}}, %{{.*}} = x86.d.pop %{{.*}} : (!x86.reg<rsp>) -> (!x86.reg<rsp>, !x86.reg)
%r_not = x86.r.not %r_pop: (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.r.not %{{.*}} : (!x86.reg) -> !x86.reg
%r_neg = x86.r.neg %r_not : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.r.neg %{{.*}} : (!x86.reg) -> !x86.reg
%r_inc = x86.r.inc %r_neg : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.r.inc %{{.*}} : (!x86.reg) -> !x86.reg
%r_dec = x86.r.dec %r_inc : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.r.dec %{{.*}} : (!x86.reg) -> !x86.reg

%r_idiv_rdx, %r_idiv_rax = x86.s.idiv %1, %rdx, %rax : (!x86.reg, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: %{{.*}}, %{{.*}} = x86.s.idiv %{{.*}}, %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
%r_imul_rdx, %r_imul_rax = x86.s.imul %1, %rax : (!x86.reg, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: %{{.*}}, %{{.*}} = x86.s.imul %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)

%rm_add_no_offset  = x86.rm.add %r_dec, %2 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rm.add %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%rm_add = x86.rm.add %rm_add_no_offset, %1, 8 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rm.add %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> !x86.reg
%rm_sub = x86.rm.sub %rm_add, %1, -8 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rm.sub %{{.*}}, %{{.*}}, -8 : (!x86.reg, !x86.reg) -> !x86.reg
%rm_imul = x86.rm.imul %rm_sub, %1, 8 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rm.imul %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> !x86.reg
%rm_and = x86.rm.and %rm_imul, %1, 8 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rm.and %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> !x86.reg
%rm_or = x86.rm.or %rm_and, %1, 8 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rm.or %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> !x86.reg
%rm_xor = x86.rm.xor %rm_or, %1, 8 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rm.xor %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> !x86.reg
%rm_mov = x86.dm.mov %1, 8 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.dm.mov %{{.*}}, 8 : (!x86.reg) -> !x86.reg
%rm_cmp = x86.sm.cmp %1, %1, 8 : (!x86.reg, !x86.reg) -> !x86.rflags<rflags>
// CHECK-NEXT: %{{.*}} = x86.sm.cmp %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> !x86.rflags
%dm_lea = x86.dm.lea %1, 8 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.dm.lea %{{.*}}, 8 : (!x86.reg) -> !x86.reg

%ri_add = x86.ri.add %rm_xor, 2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.ri.add %{{.*}}, 2 : (!x86.reg) -> !x86.reg
%ri_sub = x86.ri.sub %ri_add, 2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.ri.sub %{{.*}}, 2 : (!x86.reg) -> !x86.reg
%ri_and = x86.ri.and %ri_sub, 2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.ri.and %{{.*}}, 2 : (!x86.reg) -> !x86.reg
%ri_or = x86.ri.or %ri_and, 2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.ri.or %{{.*}}, 2 : (!x86.reg) -> !x86.reg
%ri_xor = x86.ri.xor %ri_or, 2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.ri.xor %{{.*}}, 2 : (!x86.reg) -> !x86.reg
%di_mov = x86.di.mov 2 : () -> !x86.reg
// CHECK-NEXT: %di_mov = x86.di.mov 2 : () -> !x86.reg
%ri_cmp = x86.si.cmp %1, 2 : (!x86.reg) -> !x86.rflags<rflags>
// CHECK-NEXT: %{{.*}} = x86.si.cmp %{{.*}}, 2 : (!x86.reg) -> !x86.rflags

x86.ms.add %1, %1 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.ms.add %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> ()
x86.ms.add %1, %1, 8 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.ms.add %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> ()
x86.ms.sub %1, %1, -8 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.ms.sub %{{.*}}, %{{.*}}, -8 : (!x86.reg, !x86.reg) -> ()
x86.ms.and %1, %1, 8 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.ms.and %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> ()
x86.ms.or %1, %1, 8 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.ms.or %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> ()
x86.ms.xor %1, %1, 8 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.ms.xor %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> ()
x86.ms.mov %1, %1, 8 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.ms.mov %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> ()
%mr.cmp = x86.ms.cmp %1, %1, 8 : (!x86.reg, !x86.reg) -> !x86.rflags<rflags>
// CHECK-NEXT: %{{.*}} = x86.ms.cmp %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> !x86.rflags

x86.mi.add %1, 2 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.add %{{.*}}, 2 : (!x86.reg) -> ()
x86.mi.add %1, 2, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.add %{{.*}}, 2, 8 : (!x86.reg) -> ()
x86.mi.sub %1, 2, -8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.sub %{{.*}}, 2, -8 : (!x86.reg) -> ()
x86.mi.and %1, 2, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.and %{{.*}}, 2, 8 : (!x86.reg) -> ()
x86.mi.or %1, 2, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.or %{{.*}}, 2, 8 : (!x86.reg) -> ()
x86.mi.xor %1, 2, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.xor %{{.*}}, 2, 8 : (!x86.reg) -> ()
x86.mi.mov %1, 2, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.mov %{{.*}}, 2, 8 : (!x86.reg) -> ()
%mi_cmp = x86.mi.cmp %1, 2, 8 : (!x86.reg) -> !x86.rflags<rflags>
// CHECK-NEXT: %{{.*}} = x86.mi.cmp %{{.*}}, 2, 8 : (!x86.reg) -> !x86.rflags

%rri_imul = x86.dsi.imul %1, 2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.dsi.imul %{{.*}}, 2 : (!x86.reg) -> !x86.reg

%rmi_imul_no_offset = x86.dmi.imul %1, 2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.dmi.imul %{{.*}}, 2 : (!x86.reg) -> !x86.reg
%rmi_imul = x86.dmi.imul %1, 2, 8 : (!x86.reg) ->  !x86.reg
// CHECK-NEXT: %{{.*}} = x86.dmi.imul %{{.*}}, 2, 8 : (!x86.reg) -> !x86.reg

%m_push_rsp = x86.m.push %rsp, %1 : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
// CHECK-NEXT: %{{.*}} = x86.m.push %rsp, %{{.*}} : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
%m_push_rsp2 = x86.m.push %rsp, %1, 8 : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
// CHECK-NEXT: %{{.*}} = x86.m.push %rsp, %{{.*}}, 8 : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
%m_pop_rsp = x86.m.pop %rsp, %1, 8 : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
// CHECK-NEXT: %{{.*}} = x86.m.pop %rsp, %{{.*}}, 8 : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
x86.m.neg %1 : (!x86.reg) -> ()
// CHECK-NEXT: x86.m.neg %{{.*}} : (!x86.reg) -> ()
x86.m.neg %1, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.m.neg %{{.*}}, 8 : (!x86.reg) -> ()
x86.m.not %1, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.m.not %{{.*}}, 8 : (!x86.reg) -> ()
x86.m.inc %1, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.m.inc %{{.*}}, 8 : (!x86.reg) -> ()
x86.m.dec %1, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.m.dec %{{.*}}, 8 : (!x86.reg) -> ()

%m_idiv_rdx, %m_idiv_rax = x86.m.idiv %1, %rdx, %rax : (!x86.reg, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: %{{.*}}, %{{.*}} = x86.m.idiv %{{.*}}, %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
%m_idiv_rdx2, %m_idiv_rax2 = x86.m.idiv %1, %rdx, %rax, 8 : (!x86.reg, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: %{{.*}}, %{{.*}} = x86.m.idiv %{{.*}}, %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
%m_imul_rdx, %m_imul_rax = x86.m.imul %1, %rax, 8 : (!x86.reg, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: %{{.*}}, %{{.*}} = x86.m.imul %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)

x86.directive ".text"
// CHECK-NEXT: x86.directive ".text"
x86.directive ".align" "2"
// CHECK-NEXT: x86.directive ".align" "2"
x86.label "label"
// CHECK-NEXT: x86.label "label"

func.func @funcyasm() {
    %3, %4 = "test.op"() : () -> (!x86.reg, !x86.reg)
    %rflags = x86.ss.cmp %3, %4 : (!x86.reg, !x86.reg) -> !x86.rflags<rflags>
    // CHECK: %{{.*}} = x86.ss.cmp %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.rflags

    x86.fallthrough ^fallthrough()
    // CHECK-NEXT: x86.fallthrough ^fallthrough()
    ^fallthrough:
    // CHECK-NEXT: ^fallthrough:
    x86.label "fallthrough"
    // CHECK-NEXT: x86.label "fallthrough"
    x86.c.jmp ^then(%arg : !x86.reg)
    // CHECK-NEXT: x86.c.jmp ^{{.+}}(%arg : !x86.reg)
    ^then(%arg : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg : !x86.reg):
    x86.label "then"
    // CHECK-NEXT: x86.label "then"
    x86.c.ja %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else(%arg2 : !x86.reg)
    // CHECK-NEXT: x86.c.ja %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg2 : !x86.reg)
    ^else(%arg2 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg2 : !x86.reg):
    x86.label "else"
    // CHECK-NEXT: x86.label "else"
    x86.c.jae %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else2(%arg3 : !x86.reg)
    // CHECK-NEXT: x86.c.jae %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg3 : !x86.reg)
    ^else2(%arg3 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg3 : !x86.reg):
    x86.label "else2"
    // CHECK-NEXT: x86.label "else2"
    x86.c.jb %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else3(%arg4 : !x86.reg)
    // CHECK-NEXT: x86.c.jb %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg4 : !x86.reg)
    ^else3(%arg4 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg4 : !x86.reg):
    x86.label "else3"
    // CHECK-NEXT: x86.label "else3"
    x86.c.jbe %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else4(%arg5 : !x86.reg)
    // CHECK-NEXT: x86.c.jbe %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg5 : !x86.reg)
    ^else4(%arg5 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg5 : !x86.reg):
    x86.label "else4"
    // CHECK-NEXT: x86.label "else4"
    x86.c.jc %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else5(%arg6 : !x86.reg)
    // CHECK-NEXT: x86.c.jc %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg6 : !x86.reg)
    ^else5(%arg6 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg6 : !x86.reg):
    x86.label "else5"
    // CHECK-NEXT: x86.label "else5"
    x86.c.je %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else6(%arg7 : !x86.reg)
    // CHECK-NEXT: x86.c.je %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg7 : !x86.reg)
    ^else6(%arg7 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg7 : !x86.reg):
    x86.label "else6"
    // CHECK-NEXT: x86.label "else6"
    x86.c.jg %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else7(%arg8 : !x86.reg)
    // CHECK-NEXT: x86.c.jg %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg8 : !x86.reg)
    ^else7(%arg8 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg8 : !x86.reg):
    x86.label "else7"
    // CHECK-NEXT: x86.label "else7"
    x86.c.jge %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else8(%arg9 : !x86.reg)
    // CHECK-NEXT: x86.c.jge %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg9 : !x86.reg)
    ^else8(%arg9 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg9 : !x86.reg):
    x86.label "else8"
    // CHECK-NEXT: x86.label "else8"
    x86.c.jl %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else9(%arg10 : !x86.reg)
    // CHECK-NEXT: x86.c.jl %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg10 : !x86.reg)
    ^else9(%arg10 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg10 : !x86.reg):
    x86.label "else9"
    // CHECK-NEXT: x86.label "else9"
    x86.c.jle %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else10(%arg11 : !x86.reg)
    // CHECK-NEXT: x86.c.jle %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg11 : !x86.reg)
    ^else10(%arg11 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg11 : !x86.reg):
    x86.label "else10"
    // CHECK-NEXT: x86.label "else10"
    x86.c.jna %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else11(%arg12 : !x86.reg)
    // CHECK-NEXT: x86.c.jna %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg12 : !x86.reg)
    ^else11(%arg12 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg12 : !x86.reg):
    x86.label "else11"
    // CHECK-NEXT: x86.label "else11"
    x86.c.jnae %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else12(%arg13 : !x86.reg)
    // CHECK-NEXT: x86.c.jnae %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg13 : !x86.reg)
    ^else12(%arg13 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg13 : !x86.reg):
    x86.label "else12"
    // CHECK-NEXT: x86.label "else12"
    x86.c.jnb %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else13(%arg14 : !x86.reg)
    // CHECK-NEXT: x86.c.jnb %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg14 : !x86.reg)
    ^else13(%arg14 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg14 : !x86.reg):
    x86.label "else13"
    // CHECK-NEXT: x86.label "else13"
    x86.c.jnbe %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else14(%arg15 : !x86.reg)
    // CHECK-NEXT: x86.c.jnbe %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg15 : !x86.reg)
    ^else14(%arg15 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg15 : !x86.reg):
    x86.label "else14"
    // CHECK-NEXT: x86.label "else14"
    x86.c.jnc %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else15(%arg16 : !x86.reg)
    // CHECK-NEXT: x86.c.jnc %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg16 : !x86.reg)
    ^else15(%arg16 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg16 : !x86.reg):
    x86.label "else15"
    // CHECK-NEXT: x86.label "else15"
    x86.c.jne %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else16(%arg17 : !x86.reg)
    // CHECK-NEXT: x86.c.jne %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg17 : !x86.reg)
    ^else16(%arg17 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg17 : !x86.reg):
    x86.label "else16"
    // CHECK-NEXT: x86.label "else16"
    x86.c.jng %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else17(%arg18 : !x86.reg)
    // CHECK-NEXT: x86.c.jng %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg18 : !x86.reg)
    ^else17(%arg18 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg18 : !x86.reg):
    x86.label "else17"
    // CHECK-NEXT: x86.label "else17"
    x86.c.jnge %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else18(%arg19 : !x86.reg)
    // CHECK-NEXT: x86.c.jnge %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg19 : !x86.reg)
    ^else18(%arg19 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg19 : !x86.reg):
    x86.label "else18"
    // CHECK-NEXT: x86.label "else18"
    x86.c.jnl %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else19(%arg20 : !x86.reg)
    // CHECK-NEXT: x86.c.jnl %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg20 : !x86.reg)
    ^else19(%arg20 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg20 : !x86.reg):
    x86.label "else19"
    // CHECK-NEXT: x86.label "else19"
    x86.c.jnle %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else20(%arg21 : !x86.reg)
    // CHECK-NEXT: x86.c.jnle %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg21 : !x86.reg)
    ^else20(%arg21 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg21 : !x86.reg):
    x86.label "else20"
    // CHECK-NEXT: x86.label "else20"
    x86.c.jno %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else21(%arg22 : !x86.reg)
    // CHECK-NEXT: x86.c.jno %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg22 : !x86.reg)
    ^else21(%arg22 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg22 : !x86.reg):
    x86.label "else21"
    // CHECK-NEXT: x86.label "else21"
    x86.c.jnp %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else22(%arg23 : !x86.reg)
    // CHECK-NEXT: x86.c.jnp %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg23 : !x86.reg)
    ^else22(%arg23 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg23 : !x86.reg):
    x86.label "else22"
    // CHECK-NEXT: x86.label "else22"
    x86.c.jns %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else23(%arg24 : !x86.reg)
    // CHECK-NEXT: x86.c.jns %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg24 : !x86.reg)
    ^else23(%arg24 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg24 : !x86.reg):
    x86.label "else23"
    // CHECK-NEXT: x86.label "else23"
    x86.c.jnz %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else24(%arg25 : !x86.reg)
    // CHECK-NEXT: x86.c.jnz %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg25 : !x86.reg)
    ^else24(%arg25 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg25 : !x86.reg):
    x86.label "else24"
    // CHECK-NEXT: x86.label "else24"
    x86.c.jo %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else25(%arg26 : !x86.reg)
    // CHECK-NEXT: x86.c.jo %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg26 : !x86.reg)
    ^else25(%arg26 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg26 : !x86.reg):
    x86.label "else25"
    // CHECK-NEXT: x86.label "else25"
    x86.c.jp %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else26(%arg27 : !x86.reg)
    // CHECK-NEXT: x86.c.jp %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg27 : !x86.reg)
    ^else26(%arg27 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg27 : !x86.reg):
    x86.label "else26"
    // CHECK-NEXT: x86.label "else26"
    x86.c.jpe %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else27(%arg28 : !x86.reg)
    // CHECK-NEXT: x86.c.jpe %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg28 : !x86.reg)
    ^else27(%arg28 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg28 : !x86.reg):
    x86.label "else27"
    // CHECK-NEXT: x86.label "else27"
    x86.c.jpo %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else28(%arg29 : !x86.reg)
    // CHECK-NEXT: x86.c.jpo %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg29 : !x86.reg)
    ^else28(%arg29 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg29 : !x86.reg):
    x86.label "else28"
    // CHECK-NEXT: x86.label "else28"
    x86.c.js %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else29(%arg30 : !x86.reg)
    // CHECK-NEXT: x86.c.js %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg30 : !x86.reg)
    ^else29(%arg30 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg30 : !x86.reg):
    x86.label "else29"
    // CHECK-NEXT: x86.label "else29"
    x86.c.jz %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else30(%arg31 : !x86.reg)
    // CHECK-NEXT: x86.c.jz %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg31 : !x86.reg)
    ^else30(%arg31 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg31 : !x86.reg):
    x86.label "else30"
    // CHECK-NEXT: x86.label "else30"

    x86.c.jmp ^then(%arg : !x86.reg)
    // CHECK-NEXT: x86.c.jmp ^{{.+}}(%arg : !x86.reg)
}

%xmm0, %xmm1, %xmm2 = "test.op"() : () -> (!x86.ssereg, !x86.ssereg, !x86.ssereg)

%rrr_vfmadd231pd_sse = x86.rss.vfmadd231pd %xmm0, %xmm1, %xmm2 : (!x86.ssereg, !x86.ssereg, !x86.ssereg) -> !x86.ssereg
// CHECK: %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.ssereg, !x86.ssereg, !x86.ssereg) -> !x86.ssereg
%rrm_vfmadd231pd_sse = x86.rsm.vfmadd231pd %rrr_vfmadd231pd_sse, %xmm1, %1, 8 : (!x86.ssereg, !x86.ssereg, !x86.reg) -> !x86.ssereg
// CHECK: %{{.*}} = x86.rsm.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}}, 8 : (!x86.ssereg, !x86.ssereg, !x86.reg) -> !x86.ssereg
%rrm_vfmadd231pd_sse_no_offset = x86.rsm.vfmadd231pd %rrm_vfmadd231pd_sse, %xmm1, %1 : (!x86.ssereg, !x86.ssereg, !x86.reg) -> !x86.ssereg
// CHECK: %{{.*}} = x86.rsm.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.ssereg, !x86.ssereg, !x86.reg) -> !x86.ssereg
%rrm_vfmadd231ps_sse = x86.rsm.vfmadd231ps %rrm_vfmadd231pd_sse_no_offset, %xmm1, %1, 1 : (!x86.ssereg, !x86.ssereg, !x86.reg) -> !x86.ssereg
// CHECK: %{{.*}} = x86.rsm.vfmadd231ps %{{.*}}, %{{.*}}, %{{.*}}, 1 : (!x86.ssereg, !x86.ssereg, !x86.reg) -> !x86.ssereg
%dss_addpd_sse = x86.dss.addpd %xmm1, %xmm2 : (!x86.ssereg, !x86.ssereg) -> !x86.ssereg
// CHECK-NEXT: %dss_addpd_sse = x86.dss.addpd %xmm1, %xmm2 : (!x86.ssereg, !x86.ssereg) -> !x86.ssereg
%dss_addps_sse = x86.dss.addps %xmm1, %xmm2 : (!x86.ssereg, !x86.ssereg) -> !x86.ssereg
// CHECK-NEXT: %dss_addps_sse = x86.dss.addps %xmm1, %xmm2 : (!x86.ssereg, !x86.ssereg) -> !x86.ssereg
%rm_vbroadcastsd_sse = x86.dm.vbroadcastsd %1, 8 : (!x86.reg) -> !x86.ssereg
// CHECK-NEXT: %{{.*}} = x86.dm.vbroadcastsd %{{.*}}, 8 : (!x86.reg) -> !x86.ssereg

%ymm0, %ymm1, %ymm2 = "test.op"() : () -> (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg)

%rrr_vfmadd231pd_avx2 = x86.rss.vfmadd231pd %ymm0, %ymm1, %ymm2 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
// CHECK: %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
%rrm_vfmadd231pd_avx2 = x86.rsm.vfmadd231pd %rrr_vfmadd231pd_avx2, %ymm1, %1, 8 : (!x86.avx2reg, !x86.avx2reg, !x86.reg) -> !x86.avx2reg
// CHECK: %{{.*}} = x86.rsm.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}}, 8 : (!x86.avx2reg, !x86.avx2reg, !x86.reg) -> !x86.avx2reg
%rrm_vfmadd231pd_avx2_no_offset = x86.rsm.vfmadd231pd %rrm_vfmadd231pd_avx2, %ymm1, %1 : (!x86.avx2reg, !x86.avx2reg, !x86.reg) -> !x86.avx2reg
// CHECK: %{{.*}} = x86.rsm.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx2reg, !x86.avx2reg, !x86.reg) -> !x86.avx2reg
%rrm_vfmadd231ps_avx2 = x86.rsm.vfmadd231ps %rrm_vfmadd231pd_avx2_no_offset, %ymm1, %1, 2 : (!x86.avx2reg, !x86.avx2reg, !x86.reg) -> !x86.avx2reg
// CHECK: %{{.*}} = x86.rsm.vfmadd231ps %{{.*}}, %{{.*}}, %{{.*}}, 2 : (!x86.avx2reg, !x86.avx2reg, !x86.reg) -> !x86.avx2reg
%dss_addpd_avx2 = x86.dss.addpd %ymm1, %ymm2 : (!x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
// CHECK-NEXT: %dss_addpd_avx2 = x86.dss.addpd %ymm1, %ymm2 : (!x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
%dss_addps_avx2 = x86.dss.addps %ymm1, %ymm2 : (!x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
// CHECK-NEXT: %dss_addps_avx2 = x86.dss.addps %ymm1, %ymm2 : (!x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
%rm_vbroadcastsd_avx2 = x86.dm.vbroadcastsd %1, 8 : (!x86.reg) -> !x86.avx2reg
// CHECK-NEXT: %{{.*}} = x86.dm.vbroadcastsd %{{.*}}, 8 : (!x86.reg) -> !x86.avx2reg
%ds_vpbroadcastd_avx2 = x86.ds.vpbroadcastd %rax : (!x86.reg<rax>) -> !x86.avx2reg
// CHECK-NEXT: %{{.*}} = x86.ds.vpbroadcastd %{{.*}} : (!x86.reg<rax>) -> !x86.avx2reg
%ds_vpbroadcastq_avx2 = x86.ds.vpbroadcastq %rax : (!x86.reg<rax>) -> !x86.avx2reg
// CHECK-NEXT: %{{.*}} = x86.ds.vpbroadcastq %{{.*}} : (!x86.reg<rax>) -> !x86.avx2reg

%zmm0, %zmm1, %zmm2 = "test.op"() : () -> (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg)
%k1 = "test.op"() : () -> (!x86.avx512maskreg)

%rrr_vfmadd231pd_avx512 = x86.rss.vfmadd231pd %zmm0, %zmm1, %zmm2 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
// CHECK: %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
%rrrk_vfmadd231pd_avx512_z = x86.rssk.vfmadd231pd %rrr_vfmadd231pd_avx512, %zmm1, %zmm2, %k1 {z} : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg, !x86.avx512maskreg) -> !x86.avx512reg
// CHECK: %{{.*}} = x86.rssk.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {z} : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg, !x86.avx512maskreg) -> !x86.avx512reg
%rrrk_vfmadd231pd_avx512_no_z = x86.rssk.vfmadd231pd %rrrk_vfmadd231pd_avx512_z, %zmm1, %zmm2, %k1 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg, !x86.avx512maskreg) -> !x86.avx512reg
// CHECK: %{{.*}} = x86.rssk.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg, !x86.avx512maskreg) -> !x86.avx512reg
%rrm_vfmadd231pd_avx512 = x86.rsm.vfmadd231pd %rrrk_vfmadd231pd_avx512_no_z, %zmm1, %1, 8 : (!x86.avx512reg, !x86.avx512reg, !x86.reg) -> !x86.avx512reg
// CHECK: %{{.*}} = x86.rsm.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}}, 8 : (!x86.avx512reg, !x86.avx512reg, !x86.reg) -> !x86.avx512reg
%rrm_vfmadd231pd_avx512_no_offset = x86.rsm.vfmadd231pd %rrm_vfmadd231pd_avx512, %zmm1, %1 : (!x86.avx512reg, !x86.avx512reg, !x86.reg) -> !x86.avx512reg
// CHECK: %{{.*}} = x86.rsm.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg, !x86.avx512reg, !x86.reg) -> !x86.avx512reg
%rrm_vfmadd231ps_avx512 = x86.rsm.vfmadd231ps %rrm_vfmadd231pd_avx512_no_offset, %zmm1, %1, 6 : (!x86.avx512reg, !x86.avx512reg, !x86.reg) -> !x86.avx512reg
// CHECK: %{{.*}} = x86.rsm.vfmadd231ps %{{.*}}, %{{.*}}, %{{.*}}, 6 : (!x86.avx512reg, !x86.avx512reg, !x86.reg) -> !x86.avx512reg
%dss_addpd_avx512 = x86.dss.addpd %zmm1, %zmm2 : (!x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
// CHECK-NEXT: %dss_addpd_avx512 = x86.dss.addpd %zmm1, %zmm2 : (!x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
%dss_addps_avx512 = x86.dss.addps %zmm1, %zmm2 : (!x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
// CHECK-NEXT: %dss_addps_avx512 = x86.dss.addps %zmm1, %zmm2 : (!x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
%rm_vbroadcastsd_avx512 = x86.dm.vbroadcastsd %1, 8 : (!x86.reg) -> !x86.avx512reg
// CHECK-NEXT: %{{.*}} = x86.dm.vbroadcastsd %{{.*}}, 8 : (!x86.reg) -> !x86.avx512reg

%rm_vbroadcastss_avx512 = x86.dm.vbroadcastss %1, 8 : (!x86.reg) -> (!x86.avx512reg)
// CHECK: %{{.*}} = x86.dm.vbroadcastss %{{.*}}, 8 : (!x86.reg) -> !x86.avx512reg
%rm_vbroadcastss_avx2 = x86.dm.vbroadcastss %1, 8 : (!x86.reg) -> (!x86.avx2reg)
// CHECK-NEXT: %{{.*}} = x86.dm.vbroadcastss %{{.*}}, 8 : (!x86.reg) -> !x86.avx2reg
%rm_vbroadcastss_sse = x86.dm.vbroadcastss %1, 8 : (!x86.reg) -> (!x86.ssereg)
// CHECK-NEXT: %{{.*}} = x86.dm.vbroadcastss %{{.*}}, 8 : (!x86.reg) -> !x86.ssereg

// ---- vmovapd ----
%dm_vmovapd_sse = x86.dm.vmovapd %1 : (!x86.reg) -> !x86.ssereg
// CHECK-NEXT: %dm_vmovapd_sse = x86.dm.vmovapd %1, 0 : (!x86.reg) -> !x86.ssereg
%dm_vmovapd_avx2 = x86.dm.vmovapd %1 : (!x86.reg) -> !x86.avx2reg
// CHECK-NEXT: %dm_vmovapd_avx2 = x86.dm.vmovapd %1, 0 : (!x86.reg) -> !x86.avx2reg
%dm_vmovapd_avx512 = x86.dm.vmovapd %1 : (!x86.reg) -> !x86.avx512reg
// CHECK-NEXT: %dm_vmovapd_avx512 = x86.dm.vmovapd %1, 0 : (!x86.reg) -> !x86.avx512reg

%ds_vmovapd_sse = x86.ds.vmovapd %xmm1 : (!x86.ssereg) -> !x86.ssereg
// CHECK-NEXT: %ds_vmovapd_sse = x86.ds.vmovapd %xmm1 : (!x86.ssereg) -> !x86.ssereg
%ds_vmovapd_avx2 = x86.ds.vmovapd %ymm1 : (!x86.avx2reg) -> !x86.avx2reg
// CHECK-NEXT: %ds_vmovapd_avx2 = x86.ds.vmovapd %ymm1 : (!x86.avx2reg) -> !x86.avx2reg
%ds_vmovapd_avx512 = x86.ds.vmovapd %zmm1 : (!x86.avx512reg) -> !x86.avx512reg
// CHECK-NEXT: %ds_vmovapd_avx512 = x86.ds.vmovapd %zmm1 : (!x86.avx512reg) -> !x86.avx512reg
%ds_vmovapd_avx512_mask = x86.dsk.vmovapd %zmm1, %k1 : (!x86.avx512reg, !x86.avx512maskreg) -> !x86.avx512reg
// CHECK-NEXT: %ds_vmovapd_avx512_mask = x86.dsk.vmovapd %zmm1, %k1 : (!x86.avx512reg, !x86.avx512maskreg) -> !x86.avx512reg
%ds_vmovapd_avx512_mask_z = x86.dsk.vmovapd %zmm1, %k1 {z} : (!x86.avx512reg, !x86.avx512maskreg) -> !x86.avx512reg
// CHECK-NEXT: %ds_vmovapd_avx512_mask_z = x86.dsk.vmovapd %zmm1, %k1 {z} : (!x86.avx512reg, !x86.avx512maskreg) -> !x86.avx512reg

x86.ms.vmovapd %1, %xmm1, 8 : (!x86.reg, !x86.ssereg) -> ()
// CHECK-NEXT: x86.ms.vmovapd %1, %xmm1, 8 : (!x86.reg, !x86.ssereg) -> ()
x86.ms.vmovapd %1, %ymm1, 8 : (!x86.reg, !x86.avx2reg) -> ()
// CHECK-NEXT: x86.ms.vmovapd %1, %ymm1, 8 : (!x86.reg, !x86.avx2reg) -> ()
x86.ms.vmovapd %1, %zmm1, 8 : (!x86.reg, !x86.avx512reg) -> ()
// CHECK-NEXT: x86.ms.vmovapd %1, %zmm1, 8 : (!x86.reg, !x86.avx512reg) -> ()


// ---- vmovaps ----
%dm_vmovaps_sse = x86.dm.vmovaps %1 : (!x86.reg) -> !x86.ssereg
// CHECK-NEXT: %dm_vmovaps_sse = x86.dm.vmovaps %1, 0 : (!x86.reg) -> !x86.ssereg
%dm_vmovaps_avx2 = x86.dm.vmovaps %1 : (!x86.reg) -> !x86.avx2reg
// CHECK-NEXT: %dm_vmovaps_avx2 = x86.dm.vmovaps %1, 0 : (!x86.reg) -> !x86.avx2reg
%dm_vmovaps_avx512 = x86.dm.vmovaps %1 : (!x86.reg) -> !x86.avx512reg
// CHECK-NEXT: %dm_vmovaps_avx512 = x86.dm.vmovaps %1, 0 : (!x86.reg) -> !x86.avx512reg

%ds_vmovaps_sse = x86.ds.vmovaps %xmm1 : (!x86.ssereg) -> !x86.ssereg
// CHECK-NEXT: %ds_vmovaps_sse = x86.ds.vmovaps %xmm1 : (!x86.ssereg) -> !x86.ssereg
%ds_vmovaps_avx2 = x86.ds.vmovaps %ymm1 : (!x86.avx2reg) -> !x86.avx2reg
// CHECK-NEXT: %ds_vmovaps_avx2 = x86.ds.vmovaps %ymm1 : (!x86.avx2reg) -> !x86.avx2reg
%ds_vmovaps_avx512 = x86.ds.vmovaps %zmm1 : (!x86.avx512reg) -> !x86.avx512reg
// CHECK-NEXT: %ds_vmovaps_avx512 = x86.ds.vmovaps %zmm1 : (!x86.avx512reg) -> !x86.avx512reg

x86.ms.vmovaps %1, %xmm1, 8 : (!x86.reg, !x86.ssereg) -> ()
// CHECK-NEXT: x86.ms.vmovaps %1, %xmm1, 8 : (!x86.reg, !x86.ssereg) -> ()
x86.ms.vmovaps %1, %ymm1, 8 : (!x86.reg, !x86.avx2reg) -> ()
// CHECK-NEXT: x86.ms.vmovaps %1, %ymm1, 8 : (!x86.reg, !x86.avx2reg) -> ()
x86.ms.vmovaps %1, %zmm1, 8 : (!x86.reg, !x86.avx512reg) -> ()
// CHECK-NEXT: x86.ms.vmovaps %1, %zmm1, 8 : (!x86.reg, !x86.avx512reg) -> ()


// ---- vmovupd ----
%dm_vmovupd_sse = x86.dm.vmovupd %1 : (!x86.reg) -> !x86.ssereg
// CHECK-NEXT: %dm_vmovupd_sse = x86.dm.vmovupd %1, 0 : (!x86.reg) -> !x86.ssereg
%dm_vmovupd_avx2 = x86.dm.vmovupd %1 : (!x86.reg) -> !x86.avx2reg
// CHECK-NEXT: %dm_vmovupd_avx2 = x86.dm.vmovupd %1, 0 : (!x86.reg) -> !x86.avx2reg
%dm_vmovupd_avx512 = x86.dm.vmovupd %1 : (!x86.reg) -> !x86.avx512reg
// CHECK-NEXT: %dm_vmovupd_avx512 = x86.dm.vmovupd %1, 0 : (!x86.reg) -> !x86.avx512reg

x86.ms.vmovupd %1, %xmm1, 8 : (!x86.reg, !x86.ssereg) -> ()
// CHECK-NEXT: x86.ms.vmovupd %1, %xmm1, 8 : (!x86.reg, !x86.ssereg) -> ()
x86.ms.vmovupd %1, %ymm1, 8 : (!x86.reg, !x86.avx2reg) -> ()
// CHECK-NEXT: x86.ms.vmovupd %1, %ymm1, 8 : (!x86.reg, !x86.avx2reg) -> ()
x86.ms.vmovupd %1, %zmm1, 8 : (!x86.reg, !x86.avx512reg) -> ()
// CHECK-NEXT: x86.ms.vmovupd %1, %zmm1, 8 : (!x86.reg, !x86.avx512reg) -> ()


// ---- vmovups ----
%dm_vmovups_sse = x86.dm.vmovups %1 : (!x86.reg) -> !x86.ssereg
// CHECK-NEXT: %dm_vmovups_sse = x86.dm.vmovups %1, 0 : (!x86.reg) -> !x86.ssereg
%dm_vmovups_avx2 = x86.dm.vmovups %1 : (!x86.reg) -> !x86.avx2reg
// CHECK-NEXT: %dm_vmovups_avx2 = x86.dm.vmovups %1, 0 : (!x86.reg) -> !x86.avx2reg
%dm_vmovups_avx512 = x86.dm.vmovups %1 : (!x86.reg) -> !x86.avx512reg
// CHECK-NEXT: %dm_vmovups_avx512 = x86.dm.vmovups %1, 0 : (!x86.reg) -> !x86.avx512reg

x86.ms.vmovups %1, %xmm1, 8 : (!x86.reg, !x86.ssereg) -> ()
// CHECK-NEXT: x86.ms.vmovups %1, %xmm1, 8 : (!x86.reg, !x86.ssereg) -> ()
x86.ms.vmovups %1, %ymm1, 8 : (!x86.reg, !x86.avx2reg) -> ()
// CHECK-NEXT: x86.ms.vmovups %1, %ymm1, 8 : (!x86.reg, !x86.avx2reg) -> ()
x86.ms.vmovups %1, %zmm1, 8 : (!x86.reg, !x86.avx512reg) -> ()
// CHECK-NEXT: x86.ms.vmovups %1, %zmm1, 8 : (!x86.reg, !x86.avx512reg) -> ()


// ---- vmovntpd ----
x86.ms.vmovntpd %1, %xmm1, 8 : (!x86.reg, !x86.ssereg) -> ()
// CHECK-NEXT: x86.ms.vmovntpd %1, %xmm1, 8 : (!x86.reg, !x86.ssereg) -> ()
x86.ms.vmovntpd %1, %ymm1, 8 : (!x86.reg, !x86.avx2reg) -> ()
// CHECK-NEXT: x86.ms.vmovntpd %1, %ymm1, 8 : (!x86.reg, !x86.avx2reg) -> ()
x86.ms.vmovntpd %1, %zmm1, 8 : (!x86.reg, !x86.avx512reg) -> ()
// CHECK-NEXT: x86.ms.vmovntpd %1, %zmm1, 8 : (!x86.reg, !x86.avx512reg) -> ()


// ---- vmovntps ----
x86.ms.vmovntps %1, %xmm1, 8 : (!x86.reg, !x86.ssereg) -> ()
// CHECK-NEXT: x86.ms.vmovntps %1, %xmm1, 8 : (!x86.reg, !x86.ssereg) -> ()
x86.ms.vmovntps %1, %ymm1, 8 : (!x86.reg, !x86.avx2reg) -> ()
// CHECK-NEXT: x86.ms.vmovntps %1, %ymm1, 8 : (!x86.reg, !x86.avx2reg) -> ()
x86.ms.vmovntps %1, %zmm1, 8 : (!x86.reg, !x86.avx512reg) -> ()
// CHECK-NEXT: x86.ms.vmovntps %1, %zmm1, 8 : (!x86.reg, !x86.avx512reg) -> ()

%rrr_vfmadd231ps_sse = x86.rss.vfmadd231ps %ds_vmovapd_sse, %xmm1, %xmm2 : (!x86.ssereg, !x86.ssereg, !x86.ssereg) -> !x86.ssereg
// CHECK: %{{.*}} = x86.rss.vfmadd231ps %{{.*}}, %{{.*}}, %{{.*}} : (!x86.ssereg, !x86.ssereg, !x86.ssereg) -> !x86.ssereg
%rrr_vfmadd231ps_avx2 = x86.rss.vfmadd231ps %ds_vmovapd_avx2, %ymm1, %ymm2 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
// CHECK: %{{.*}} = x86.rss.vfmadd231ps %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
%rrr_vfmadd231ps_avx512 = x86.rss.vfmadd231ps %ds_vmovapd_avx512, %zmm1, %zmm2 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
// CHECK: %{{.*}} = x86.rss.vfmadd231ps %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg

%shuf_res = x86.dssi.shufps %zmm1, %zmm2, 170 : (!x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
// CHECK: %shuf_res = x86.dssi.shufps %zmm1, %zmm2, 170 : (!x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
