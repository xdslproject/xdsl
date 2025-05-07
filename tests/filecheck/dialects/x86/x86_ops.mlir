// RUN: XDSL_ROUNDTRIP
%a = "test.op"() : () -> !x86.reg<rax>
// CHECK: %{{.*}} = "test.op"() : () -> !x86.reg<rax>

%0, %1 = "test.op"() : () -> (!x86.reg, !x86.reg)
%rsp = "test.op"() : () -> !x86.reg<rsp>
%rax = "test.op"() : () -> !x86.reg<rax>
%rdx = "test.op"() : () -> !x86.reg<rdx>

%rr_add = x86.rr.add %0, %1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK: %{{.*}} = x86.rr.add %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%rr_sub = x86.rr.sub %0, %1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rr.sub %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%rr_mul = x86.rr.imul %0, %1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rr.imul %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%rr_and = x86.rr.and %0, %1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rr.and %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%rr_or = x86.rr.or %0, %1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rr.or %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%rr_xor = x86.rr.xor %0, %1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rr.xor %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%rr_mov = x86.rr.mov %0, %1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rr.mov %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%rr_cmp = x86.rr.cmp %0, %1 : (!x86.reg, !x86.reg) -> !x86.rflags<rflags>
// CHECK: %{{.*}} = x86.rr.cmp %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.rflags

%r_pushrsp = x86.r.push %rsp, %0 : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
// CHECK-NEXT: %{{.*}} = x86.r.push %rsp, %{{.*}} : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
%r_pop, %r_poprsp = x86.r.pop %rsp : (!x86.reg<rsp>) -> (!x86.reg, !x86.reg<rsp>)
// CHECK-NEXT: %{{.*}}, %{{.*}} = x86.r.pop %{{.*}} : (!x86.reg<rsp>) -> (!x86.reg, !x86.reg<rsp>)
%r_not = x86.r.not %0 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.r.not %{{.*}} : (!x86.reg) -> !x86.reg
%r_neg = x86.r.neg %0 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.r.neg %{{.*}} : (!x86.reg) -> !x86.reg
%r_inc = x86.r.inc %0 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.r.inc %{{.*}} : (!x86.reg) -> !x86.reg
%r_dec = x86.r.dec %0 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.r.dec %{{.*}} : (!x86.reg) -> !x86.reg

%r_idiv_rdx, %r_idiv_rax = x86.r.idiv %0, %rdx, %rax : (!x86.reg, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: %{{.*}}, %{{.*}} = x86.r.idiv %{{.*}}, %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
%r_imul_rdx, %r_imul_rax = x86.r.imul %0, %rax : (!x86.reg, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: %{{.*}}, %{{.*}} = x86.r.imul %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)

%rm_add_no_offset  = x86.rm.add %0, %1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rm.add %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.reg
%rm_add = x86.rm.add %0, %1, 8 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK: %{{.*}} = x86.rm.add %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> !x86.reg
%rm_sub = x86.rm.sub %0, %1, -8 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rm.sub %{{.*}}, %{{.*}}, -8 : (!x86.reg, !x86.reg) -> !x86.reg
%rm_imul = x86.rm.imul %0, %1, 8 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rm.imul %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> !x86.reg
%rm_and = x86.rm.and %0, %1, 8 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rm.and %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> !x86.reg
%rm_or = x86.rm.or %0, %1, 8 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rm.or %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> !x86.reg
%rm_xor = x86.rm.xor %0, %1, 8 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rm.xor %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> !x86.reg
%rm_mov = x86.rm.mov %1, 8 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rm.mov %{{.*}}, 8 : (!x86.reg) -> !x86.reg
%rm_cmp = x86.rm.cmp %0, %1, 8 : (!x86.reg, !x86.reg) -> !x86.rflags<rflags>
// CHECK-NEXT: %{{.*}} = x86.rm.cmp %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> !x86.rflags
%rm_lea = x86.rm.lea %0, %1, 8 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rm.lea %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> !x86.reg

%ri_add = x86.ri.add %0, 2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.ri.add %{{.*}}, 2 : (!x86.reg) -> !x86.reg
%ri_sub = x86.ri.sub %0, 2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.ri.sub %{{.*}}, 2 : (!x86.reg) -> !x86.reg
%ri_and = x86.ri.and %0, 2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.ri.and %{{.*}}, 2 : (!x86.reg) -> !x86.reg
%ri_or = x86.ri.or %0, 2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.ri.or %{{.*}}, 2 : (!x86.reg) -> !x86.reg
%ri_xor = x86.ri.xor %0, 2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.ri.xor %{{.*}}, 2 : (!x86.reg) -> !x86.reg
%ri_mov = x86.ri.mov %0, 2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.ri.mov %{{.*}}, 2 : (!x86.reg) -> !x86.reg
%ri_cmp = x86.ri.cmp %0, 2 : (!x86.reg) -> !x86.rflags<rflags>
// CHECK-NEXT: %{{.*}} = x86.ri.cmp %{{.*}}, 2 : (!x86.reg) -> !x86.rflags

x86.mr.add %0, %1 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.mr.add %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> ()
x86.mr.add %0, %1, 8 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.mr.add %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> ()
x86.mr.sub %0, %1, -8 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.mr.sub %{{.*}}, %{{.*}}, -8 : (!x86.reg, !x86.reg) -> ()
x86.mr.and %0, %1, 8 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.mr.and %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> ()
x86.mr.or %0, %1, 8 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.mr.or %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> ()
x86.mr.xor %0, %1, 8 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.mr.xor %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> ()
x86.mr.mov %0, %1, 8 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.mr.mov %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> ()
%mr.cmp = x86.mr.cmp %0, %1, 8 : (!x86.reg, !x86.reg) -> !x86.rflags<rflags>
// CHECK-NEXT: %{{.*}} = x86.mr.cmp %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg) -> !x86.rflags

x86.mi.add %0, 2 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.add %{{.*}}, 2 : (!x86.reg) -> ()
x86.mi.add %0, 2, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.add %{{.*}}, 2, 8 : (!x86.reg) -> ()
x86.mi.sub %0, 2, -8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.sub %{{.*}}, 2, -8 : (!x86.reg) -> ()
x86.mi.and %0, 2, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.and %{{.*}}, 2, 8 : (!x86.reg) -> ()
x86.mi.or %0, 2, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.or %{{.*}}, 2, 8 : (!x86.reg) -> ()
x86.mi.xor %0, 2, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.xor %{{.*}}, 2, 8 : (!x86.reg) -> ()
x86.mi.mov %0, 2, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.mov %{{.*}}, 2, 8 : (!x86.reg) -> ()
%mi_cmp = x86.mi.cmp %0, 2, 8 : (!x86.reg) -> !x86.rflags<rflags>
// CHECK-NEXT: %{{.*}} = x86.mi.cmp %{{.*}}, 2, 8 : (!x86.reg) -> !x86.rflags

%rri_imul = x86.rri.imul %1, 2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rri.imul %{{.*}}, 2 : (!x86.reg) -> !x86.reg

%rmi_imul_no_offset = x86.rmi.imul %1, 2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rmi.imul %{{.*}}, 2 : (!x86.reg) -> !x86.reg
%rmi_imul = x86.rmi.imul %1, 2, 8 : (!x86.reg) ->  !x86.reg
// CHECK-NEXT: %{{.*}} = x86.rmi.imul %{{.*}}, 2, 8 : (!x86.reg) -> !x86.reg

%m_push_rsp = x86.m.push %rsp, %0 : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
// CHECK-NEXT: %{{.*}} = x86.m.push %rsp, %{{.*}} : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
%m_push_rsp2 = x86.m.push %rsp, %0, 8 : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
// CHECK-NEXT: %{{.*}} = x86.m.push %rsp, %{{.*}}, 8 : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
%m_pop_rsp = x86.m.pop %rsp, %0, 8 : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
// CHECK-NEXT: %{{.*}} = x86.m.pop %rsp, %{{.*}}, 8 : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
x86.m.neg %0 : (!x86.reg) -> ()
// CHECK-NEXT: x86.m.neg %{{.*}} : (!x86.reg) -> ()
x86.m.neg %0, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.m.neg %{{.*}}, 8 : (!x86.reg) -> ()
x86.m.not %0, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.m.not %{{.*}}, 8 : (!x86.reg) -> ()
x86.m.inc %0, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.m.inc %{{.*}}, 8 : (!x86.reg) -> ()
x86.m.dec %0, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.m.dec %{{.*}}, 8 : (!x86.reg) -> ()

%m_idiv_rdx, %m_idiv_rax = x86.m.idiv %0, %rdx, %rax : (!x86.reg, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: %{{.*}}, %{{.*}} = x86.m.idiv %{{.*}}, %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
%m_idiv_rdx2, %m_idiv_rax2 = x86.m.idiv %0, %rdx, %rax, 8 : (!x86.reg, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: %{{.*}}, %{{.*}} = x86.m.idiv %{{.*}}, %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
%m_imul_rdx, %m_imul_rax = x86.m.imul %0, %rax, 8 : (!x86.reg, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: %{{.*}}, %{{.*}} = x86.m.imul %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)

x86.directive ".text"
// CHECK-NEXT: x86.directive ".text"
x86.directive ".align" "2"
// CHECK-NEXT: x86.directive ".align" "2"
x86.label "label"
// CHECK-NEXT: x86.label "label"

func.func @funcyasm() {
    %2, %3 = "test.op"() : () -> (!x86.reg, !x86.reg)
    %rflags = x86.rr.cmp %2, %3 : (!x86.reg, !x86.reg) -> !x86.rflags<rflags>
    // CHECK: %{{.*}} = x86.rr.cmp %{{.*}}, %{{.*}} : (!x86.reg, !x86.reg) -> !x86.rflags

    x86.s.jmp ^then(%arg : !x86.reg)
    // CHECK-NEXT: x86.s.jmp ^{{.+}}(%arg : !x86.reg)
    ^then(%arg : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg : !x86.reg):
    x86.label "then"
    // CHECK-NEXT: x86.label "then"
    x86.s.ja %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else(%arg2 : !x86.reg)
    // CHECK-NEXT: x86.s.ja %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg2 : !x86.reg)
    ^else(%arg2 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg2 : !x86.reg):
    x86.label "else"
    // CHECK-NEXT: x86.label "else"
    x86.s.jae %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else2(%arg3 : !x86.reg)
    // CHECK-NEXT: x86.s.jae %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg3 : !x86.reg)
    ^else2(%arg3 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg3 : !x86.reg):
    x86.label "else2"
    // CHECK-NEXT: x86.label "else2"
    x86.s.jb %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else3(%arg4 : !x86.reg)
    // CHECK-NEXT: x86.s.jb %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg4 : !x86.reg)
    ^else3(%arg4 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg4 : !x86.reg):
    x86.label "else3"
    // CHECK-NEXT: x86.label "else3"
    x86.s.jbe %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else4(%arg5 : !x86.reg)
    // CHECK-NEXT: x86.s.jbe %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg5 : !x86.reg)
    ^else4(%arg5 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg5 : !x86.reg):
    x86.label "else4"
    // CHECK-NEXT: x86.label "else4"
    x86.s.jc %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else5(%arg6 : !x86.reg)
    // CHECK-NEXT: x86.s.jc %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg6 : !x86.reg)
    ^else5(%arg6 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg6 : !x86.reg):
    x86.label "else5"
    // CHECK-NEXT: x86.label "else5"
    x86.s.je %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else6(%arg7 : !x86.reg)
    // CHECK-NEXT: x86.s.je %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg7 : !x86.reg)
    ^else6(%arg7 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg7 : !x86.reg):
    x86.label "else6"
    // CHECK-NEXT: x86.label "else6"
    x86.s.jg %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else7(%arg8 : !x86.reg)
    // CHECK-NEXT: x86.s.jg %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg8 : !x86.reg)
    ^else7(%arg8 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg8 : !x86.reg):
    x86.label "else7"
    // CHECK-NEXT: x86.label "else7"
    x86.s.jge %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else8(%arg9 : !x86.reg)
    // CHECK-NEXT: x86.s.jge %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg9 : !x86.reg)
    ^else8(%arg9 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg9 : !x86.reg):
    x86.label "else8"
    // CHECK-NEXT: x86.label "else8"
    x86.s.jl %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else9(%arg10 : !x86.reg)
    // CHECK-NEXT: x86.s.jl %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg10 : !x86.reg)
    ^else9(%arg10 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg10 : !x86.reg):
    x86.label "else9"
    // CHECK-NEXT: x86.label "else9"
    x86.s.jle %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else10(%arg11 : !x86.reg)
    // CHECK-NEXT: x86.s.jle %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg11 : !x86.reg)
    ^else10(%arg11 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg11 : !x86.reg):
    x86.label "else10"
    // CHECK-NEXT: x86.label "else10"
    x86.s.jna %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else11(%arg12 : !x86.reg)
    // CHECK-NEXT: x86.s.jna %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg12 : !x86.reg)
    ^else11(%arg12 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg12 : !x86.reg):
    x86.label "else11"
    // CHECK-NEXT: x86.label "else11"
    x86.s.jnae %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else12(%arg13 : !x86.reg)
    // CHECK-NEXT: x86.s.jnae %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg13 : !x86.reg)
    ^else12(%arg13 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg13 : !x86.reg):
    x86.label "else12"
    // CHECK-NEXT: x86.label "else12"
    x86.s.jnb %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else13(%arg14 : !x86.reg)
    // CHECK-NEXT: x86.s.jnb %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg14 : !x86.reg)
    ^else13(%arg14 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg14 : !x86.reg):
    x86.label "else13"
    // CHECK-NEXT: x86.label "else13"
    x86.s.jnbe %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else14(%arg15 : !x86.reg)
    // CHECK-NEXT: x86.s.jnbe %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg15 : !x86.reg)
    ^else14(%arg15 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg15 : !x86.reg):
    x86.label "else14"
    // CHECK-NEXT: x86.label "else14"
    x86.s.jnc %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else15(%arg16 : !x86.reg)
    // CHECK-NEXT: x86.s.jnc %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg16 : !x86.reg)
    ^else15(%arg16 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg16 : !x86.reg):
    x86.label "else15"
    // CHECK-NEXT: x86.label "else15"
    x86.s.jne %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else16(%arg17 : !x86.reg)
    // CHECK-NEXT: x86.s.jne %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg17 : !x86.reg)
    ^else16(%arg17 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg17 : !x86.reg):
    x86.label "else16"
    // CHECK-NEXT: x86.label "else16"
    x86.s.jng %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else17(%arg18 : !x86.reg)
    // CHECK-NEXT: x86.s.jng %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg18 : !x86.reg)
    ^else17(%arg18 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg18 : !x86.reg):
    x86.label "else17"
    // CHECK-NEXT: x86.label "else17"
    x86.s.jnge %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else18(%arg19 : !x86.reg)
    // CHECK-NEXT: x86.s.jnge %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg19 : !x86.reg)
    ^else18(%arg19 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg19 : !x86.reg):
    x86.label "else18"
    // CHECK-NEXT: x86.label "else18"
    x86.s.jnl %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else19(%arg20 : !x86.reg)
    // CHECK-NEXT: x86.s.jnl %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg20 : !x86.reg)
    ^else19(%arg20 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg20 : !x86.reg):
    x86.label "else19"
    // CHECK-NEXT: x86.label "else19"
    x86.s.jnle %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else20(%arg21 : !x86.reg)
    // CHECK-NEXT: x86.s.jnle %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg21 : !x86.reg)
    ^else20(%arg21 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg21 : !x86.reg):
    x86.label "else20"
    // CHECK-NEXT: x86.label "else20"
    x86.s.jno %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else21(%arg22 : !x86.reg)
    // CHECK-NEXT: x86.s.jno %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg22 : !x86.reg)
    ^else21(%arg22 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg22 : !x86.reg):
    x86.label "else21"
    // CHECK-NEXT: x86.label "else21"
    x86.s.jnp %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else22(%arg23 : !x86.reg)
    // CHECK-NEXT: x86.s.jnp %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg23 : !x86.reg)
    ^else22(%arg23 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg23 : !x86.reg):
    x86.label "else22"
    // CHECK-NEXT: x86.label "else22"
    x86.s.jns %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else23(%arg24 : !x86.reg)
    // CHECK-NEXT: x86.s.jns %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg24 : !x86.reg)
    ^else23(%arg24 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg24 : !x86.reg):
    x86.label "else23"
    // CHECK-NEXT: x86.label "else23"
    x86.s.jnz %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else24(%arg25 : !x86.reg)
    // CHECK-NEXT: x86.s.jnz %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg25 : !x86.reg)
    ^else24(%arg25 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg25 : !x86.reg):
    x86.label "else24"
    // CHECK-NEXT: x86.label "else24"
    x86.s.jo %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else25(%arg26 : !x86.reg)
    // CHECK-NEXT: x86.s.jo %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg26 : !x86.reg)
    ^else25(%arg26 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg26 : !x86.reg):
    x86.label "else25"
    // CHECK-NEXT: x86.label "else25"
    x86.s.jp %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else26(%arg27 : !x86.reg)
    // CHECK-NEXT: x86.s.jp %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg27 : !x86.reg)
    ^else26(%arg27 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg27 : !x86.reg):
    x86.label "else26"
    // CHECK-NEXT: x86.label "else26"
    x86.s.jpe %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else27(%arg28 : !x86.reg)
    // CHECK-NEXT: x86.s.jpe %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg28 : !x86.reg)
    ^else27(%arg28 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg28 : !x86.reg):
    x86.label "else27"
    // CHECK-NEXT: x86.label "else27"
    x86.s.jpo %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else28(%arg29 : !x86.reg)
    // CHECK-NEXT: x86.s.jpo %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg29 : !x86.reg)
    ^else28(%arg29 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg29 : !x86.reg):
    x86.label "else28"
    // CHECK-NEXT: x86.label "else28"
    x86.s.js %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else29(%arg30 : !x86.reg)
    // CHECK-NEXT: x86.s.js %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg30 : !x86.reg)
    ^else29(%arg30 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg30 : !x86.reg):
    x86.label "else29"
    // CHECK-NEXT: x86.label "else29"
    x86.s.jz %rflags : !x86.rflags<rflags>, ^then(%arg : !x86.reg), ^else30(%arg31 : !x86.reg)
    // CHECK-NEXT: x86.s.jz %rflags : !x86.rflags<rflags>, ^{{.+}}(%arg : !x86.reg), ^{{.+}}(%arg31 : !x86.reg)
    ^else30(%arg31 : !x86.reg):
    // CHECK-NEXT: ^{{.+}}(%arg31 : !x86.reg):
    x86.label "else30"
    // CHECK-NEXT: x86.label "else30"

    x86.s.jmp ^then(%arg : !x86.reg)
    // CHECK-NEXT: x86.s.jmp ^{{.+}}(%arg : !x86.reg)
}

%xmm0, %xmm1, %xmm2 = "test.op"() : () -> (!x86.ssereg, !x86.ssereg, !x86.ssereg)

%rrr_vfmadd231pd_sse = x86.rrr.vfmadd231pd %xmm0, %xmm1, %xmm2 : (!x86.ssereg, !x86.ssereg, !x86.ssereg) -> !x86.ssereg
// CHECK: %{{.*}} = x86.rrr.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.ssereg, !x86.ssereg, !x86.ssereg) -> !x86.ssereg
%rr_vmovapd_sse = x86.rr.vmovapd %xmm0, %xmm1 : (!x86.ssereg, !x86.ssereg) -> !x86.ssereg
// CHECK-NEXT: x86.rr.vmovapd %{{.*}}, %{{.*}} : (!x86.ssereg, !x86.ssereg) -> !x86.ssereg
x86.mr.vmovapd %0, %xmm1, 8 : (!x86.reg, !x86.ssereg) -> ()
// CHECK-NEXT: x86.mr.vmovapd %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.ssereg) -> ()
%rm_vbroadcastsd_sse = x86.rm.vbroadcastsd %0, 8 : (!x86.reg) -> !x86.ssereg
// CHECK-NEXT: %{{.*}} = x86.rm.vbroadcastsd %{{.*}}, 8 : (!x86.reg) -> !x86.ssereg

%ymm0, %ymm1, %ymm2 = "test.op"() : () -> (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg)

%rrr_vfmadd231pd_avx2 = x86.rrr.vfmadd231pd %ymm0, %ymm1, %ymm2 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
// CHECK: %{{.*}} = x86.rrr.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
%rr_vmovapd_avx2 = x86.rr.vmovapd %ymm0, %ymm1 : (!x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
// CHECK-NEXT: x86.rr.vmovapd %{{.*}}, %{{.*}} : (!x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
x86.mr.vmovapd %0, %ymm1, 8 : (!x86.reg, !x86.avx2reg) -> ()
// CHECK-NEXT: x86.mr.vmovapd %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.avx2reg) -> ()
%rm_vbroadcastsd_avx2 = x86.rm.vbroadcastsd %0, 8 : (!x86.reg) -> !x86.avx2reg
// CHECK-NEXT: %{{.*}} = x86.rm.vbroadcastsd %{{.*}}, 8 : (!x86.reg) -> !x86.avx2reg

%zmm0, %zmm1, %zmm2 = "test.op"() : () -> (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg)

%rrr_vfmadd231pd_avx512 = x86.rrr.vfmadd231pd %zmm0, %zmm1, %zmm2 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
// CHECK: %{{.*}} = x86.rrr.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
%rr_vmovapd_avx512 = x86.rr.vmovapd %zmm0, %zmm1 : (!x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
// CHECK-NEXT: x86.rr.vmovapd %{{.*}}, %{{.*}} : (!x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
x86.mr.vmovapd %0, %zmm1, 8 : (!x86.reg, !x86.avx512reg) -> ()
// CHECK-NEXT: x86.mr.vmovapd %{{.*}}, %{{.*}}, 8 : (!x86.reg, !x86.avx512reg) -> ()
%rm_vbroadcastsd_avx512 = x86.rm.vbroadcastsd %0, 8 : (!x86.reg) -> !x86.avx512reg
// CHECK-NEXT: %{{.*}} = x86.rm.vbroadcastsd %{{.*}}, 8 : (!x86.reg) -> !x86.avx512reg

%rm_vmovups_avx512 = x86.rm.vmovups %1, 8 : (!x86.reg) -> (!x86.avx512reg)
// CHECK: %{{.*}} = x86.rm.vmovups %{{.*}}, 8 : (!x86.reg) -> !x86.avx512reg
%rm_vmovups_avx2 = x86.rm.vmovups %1, 8 : (!x86.reg) -> (!x86.avx2reg)
// CHECK-NEXT: %{{.*}} = x86.rm.vmovups %{{.*}}, 8 : (!x86.reg) -> !x86.avx2reg
%rm_vmovups_sse = x86.rm.vmovups %1, 8 : (!x86.reg) -> (!x86.ssereg)
// CHECK-NEXT: %{{.*}} = x86.rm.vmovups %{{.*}}, 8 : (!x86.reg) -> !x86.ssereg

%rm_vbroadcastss_avx512 = x86.rm.vbroadcastss %1, 8 : (!x86.reg) -> (!x86.avx512reg)
// CHECK: %{{.*}} = x86.rm.vbroadcastss %{{.*}}, 8 : (!x86.reg) -> !x86.avx512reg
%rm_vbroadcastss_avx2 = x86.rm.vbroadcastss %1, 8 : (!x86.reg) -> (!x86.avx2reg)
// CHECK-NEXT: %{{.*}} = x86.rm.vbroadcastss %{{.*}}, 8 : (!x86.reg) -> !x86.avx2reg
%rm_vbroadcastss_sse = x86.rm.vbroadcastss %1, 8 : (!x86.reg) -> (!x86.ssereg)
// CHECK-NEXT: %{{.*}} = x86.rm.vbroadcastss %{{.*}}, 8 : (!x86.reg) -> !x86.ssereg

x86.mr.vmovups %0, %zmm1, 0 : (!x86.reg, !x86.avx512reg) -> ()
// CHECK: x86.mr.vmovups %{{.*}}, %{{.*}} : (!x86.reg, !x86.avx512reg) -> ()
x86.mr.vmovups %0, %ymm1, 0 : (!x86.reg, !x86.avx2reg) -> ()
// CHECK-NEXT: x86.mr.vmovups %{{.*}}, %{{.*}} : (!x86.reg, !x86.avx2reg) -> ()
x86.mr.vmovups %0, %xmm1, 0 : (!x86.reg, !x86.ssereg) -> ()
// CHECK-NEXT: x86.mr.vmovups %{{.*}}, %{{.*}} : (!x86.reg, !x86.ssereg) -> ()

%rrr_vfmadd231ps_sse = x86.rrr.vfmadd231ps %xmm0, %xmm1, %xmm2 : (!x86.ssereg, !x86.ssereg, !x86.ssereg) -> !x86.ssereg
// CHECK: %{{.*}} = x86.rrr.vfmadd231ps %{{.*}}, %{{.*}}, %{{.*}} : (!x86.ssereg, !x86.ssereg, !x86.ssereg) -> !x86.ssereg
%rrr_vfmadd231ps_avx2 = x86.rrr.vfmadd231ps %ymm0, %ymm1, %ymm2 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
// CHECK: %{{.*}} = x86.rrr.vfmadd231ps %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
%rrr_vfmadd231ps_avx512 = x86.rrr.vfmadd231ps %zmm0, %zmm1, %zmm2 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
// CHECK: %{{.*}} = x86.rrr.vfmadd231ps %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
