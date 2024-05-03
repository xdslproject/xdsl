// RUN: XDSL_ROUNDTRIP
%a = "test.op"() : () -> !x86.reg<rax>
// CHECK: %{{.*}} = "test.op"() : () -> !x86.reg<rax>

%0, %1 = "test.op"() : () -> (!x86.reg<>, !x86.reg<>)
%rsp = "test.op"() : () -> !x86.reg<rsp>
%rax = "test.op"() : () -> !x86.reg<rax>
%rdx = "test.op"() : () -> !x86.reg<rdx>

%add = x86.rr.add %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK: %{{.*}} = x86.rr.add %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%sub = x86.rr.sub %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rr.sub %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%mul = x86.rr.imul %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rr.imul %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%and = x86.rr.and %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rr.and %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%or = x86.rr.or %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rr.or %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%xor = x86.rr.xor %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rr.xor %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%mov = x86.rr.mov %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rr.mov %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
x86.r.push %0 : (!x86.reg<>) -> ()
// CHECK-NEXT: x86.r.push %{{.*}} : (!x86.reg<>)
%pop, %poprsp = x86.r.pop %rsp : (!x86.reg<rsp>) -> (!x86.reg<>, !x86.reg<rsp>)
// CHECK-NEXT: %{{.*}}, %{{.*}} = x86.r.pop %{{.*}} : (!x86.reg<rsp>) -> (!x86.reg<>, !x86.reg<rsp>)
%not = x86.r.not %0 : (!x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.r.not %{{.*}} : (!x86.reg<>) -> !x86.reg<>

%r_idiv_rdx, %r_idiv_rax = x86.r.idiv %0, %rdx, %rax : (!x86.reg<>, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)
// CHECK-NEXT: %{{.*}}, %{{.*}} = x86.r.idiv %{{.*}}, %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<rdx>, !x86.reg<rax>) -> (!x86.reg<rdx>, !x86.reg<rax>)

%rm_add_no_offset  = x86.rm.add %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rm.add %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%rm_add = x86.rm.add %0, %1, 8 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK: %{{.*}} = x86.rm.add %{{.*}}, %{{.*}}, 8 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%rm_sub = x86.rm.sub %0, %1, -8 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rm.sub %{{.*}}, %{{.*}}, -8 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%rm_imul = x86.rm.imul %0, %1, 8 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rm.imul %{{.*}}, %{{.*}}, 8 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%rm_and = x86.rm.and %0, %1, 8 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rm.and %{{.*}}, %{{.*}}, 8 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%rm_or = x86.rm.or %0, %1, 8 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rm.or %{{.*}}, %{{.*}}, 8 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%rm_xor = x86.rm.xor %0, %1, 8 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rm.xor %{{.*}}, %{{.*}}, 8 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%rm_mov = x86.rm.mov %0, %1, 8 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rm.mov %{{.*}}, %{{.*}}, 8 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>

%ri_add = x86.ri.add %0, 2 : (!x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.ri.add %{{.*}}, 2 : (!x86.reg<>) -> !x86.reg<>
%ri_sub = x86.ri.sub %0, 2 : (!x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.ri.sub %{{.*}}, 2 : (!x86.reg<>) -> !x86.reg<>
%ri_and = x86.ri.and %0, 2 : (!x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.ri.and %{{.*}}, 2 : (!x86.reg<>) -> !x86.reg<>
%ri_or = x86.ri.or %0, 2 : (!x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.ri.or %{{.*}}, 2 : (!x86.reg<>) -> !x86.reg<>
%ri_xor = x86.ri.xor %0, 2 : (!x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.ri.xor %{{.*}}, 2 : (!x86.reg<>) -> !x86.reg<>
%ri_mov = x86.ri.mov %0, 2 : (!x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.ri.mov %{{.*}}, 2 : (!x86.reg<>) -> !x86.reg<>

x86.mr.add %0, %1 : (!x86.reg<>, !x86.reg<>) -> ()
// CHECK-NEXT: x86.mr.add %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> ()
x86.mr.add %0, %1, 8 : (!x86.reg<>, !x86.reg<>) -> ()
// CHECK-NEXT: x86.mr.add %{{.*}}, %{{.*}}, 8 : (!x86.reg<>, !x86.reg<>) -> ()
x86.mr.sub %0, %1, -8 : (!x86.reg<>, !x86.reg<>) -> ()
// CHECK-NEXT: x86.mr.sub %{{.*}}, %{{.*}}, -8 : (!x86.reg<>, !x86.reg<>) -> ()
x86.mr.and %0, %1, 8 : (!x86.reg<>, !x86.reg<>) -> ()
// CHECK-NEXT: x86.mr.and %{{.*}}, %{{.*}}, 8 : (!x86.reg<>, !x86.reg<>) -> ()
x86.mr.or %0, %1, 8 : (!x86.reg<>, !x86.reg<>) -> ()
// CHECK-NEXT: x86.mr.or %{{.*}}, %{{.*}}, 8 : (!x86.reg<>, !x86.reg<>) -> ()
x86.mr.xor %0, %1, 8 : (!x86.reg<>, !x86.reg<>) -> ()
// CHECK-NEXT: x86.mr.xor %{{.*}}, %{{.*}}, 8 : (!x86.reg<>, !x86.reg<>) -> ()
x86.mr.mov %0, %1, 8 : (!x86.reg<>, !x86.reg<>) -> ()
// CHECK-NEXT: x86.mr.mov %{{.*}}, %{{.*}}, 8 : (!x86.reg<>, !x86.reg<>) -> ()

x86.mi.add %0, 2 : (!x86.reg<>) -> ()
// CHECK-NEXT: x86.mi.add %{{.*}}, 2 : (!x86.reg<>) -> ()
x86.mi.add %0, 2, 8 : (!x86.reg<>) -> ()
// CHECK-NEXT: x86.mi.add %{{.*}}, 2, 8 : (!x86.reg<>) -> ()
x86.mi.sub %0, 2, -8 : (!x86.reg<>) -> ()
// CHECK-NEXT: x86.mi.sub %{{.*}}, 2, -8 : (!x86.reg<>) -> ()
x86.mi.and %0, 2, 8 : (!x86.reg<>) -> ()
// CHECK-NEXT: x86.mi.and %{{.*}}, 2, 8 : (!x86.reg<>) -> ()
x86.mi.or %0, 2, 8 : (!x86.reg<>) -> ()
// CHECK-NEXT: x86.mi.or %{{.*}}, 2, 8 : (!x86.reg<>) -> ()
x86.mi.xor %0, 2, 8 : (!x86.reg<>) -> ()
// CHECK-NEXT: x86.mi.xor %{{.*}}, 2, 8 : (!x86.reg<>) -> ()
x86.mi.mov %0, 2, 8 : (!x86.reg<>) -> ()
// CHECK-NEXT: x86.mi.mov %{{.*}}, 2, 8 : (!x86.reg<>) -> ()

%rri_imul = x86.rri.imul %1, 2 : (!x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rri.imul %{{.*}}, 2 : (!x86.reg<>) -> !x86.reg<>

%rmi_imul_no_offset = x86.rmi.imul %1, 2 : (!x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rmi.imul %{{.*}}, 2 : (!x86.reg<>) -> !x86.reg<>
%rmi_imul = x86.rmi.imul %1, 2, 8 : (!x86.reg<>) ->  !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rmi.imul %{{.*}}, 2, 8 : (!x86.reg<>) -> !x86.reg<>

x86.m.push %0 : (!x86.reg<>) -> ()
// CHECK-NEXT: x86.m.push %{{.*}} : (!x86.reg<>) -> ()
x86.m.push %0, 8 : (!x86.reg<>) -> ()
// CHECK-NEXT: x86.m.push %{{.*}}, 8 : (!x86.reg<>) -> ()
x86.m.neg %0 : (!x86.reg<>) -> ()
// CHECK-NEXT: x86.m.neg %{{.*}} : (!x86.reg<>) -> ()
x86.m.neg %0, 8 : (!x86.reg<>) -> ()
// CHECK-NEXT: x86.m.neg %{{.*}}, 8 : (!x86.reg<>) -> ()
x86.m.not %0, 8 : (!x86.reg<>) -> ()
// CHECK-NEXT: x86.m.not %{{.*}}, 8 : (!x86.reg<>) -> ()