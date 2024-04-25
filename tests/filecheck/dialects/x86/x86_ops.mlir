// RUN: XDSL_ROUNDTRIP
%a = "test.op"() : () -> !x86.reg<rax>
// CHECK: %{{.*}} = "test.op"() : () -> !x86.reg<rax>

%0, %1 = "test.op"() : () -> (!x86.reg<>, !x86.reg<>)
%rsp = "test.op"() : () -> !x86.reg<rsp>

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
%ri_imul = x86.ri.imul %0, 2 : (!x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.ri.imul %{{.*}}, 2 : (!x86.reg<>) -> !x86.reg<>
%ri_and = x86.ri.and %0, 2 : (!x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.ri.and %{{.*}}, 2 : (!x86.reg<>) -> !x86.reg<>
%ri_or = x86.ri.or %0, 2 : (!x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.ri.or %{{.*}}, 2 : (!x86.reg<>) -> !x86.reg<>
%ri_xor = x86.ri.xor %0, 2 : (!x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.ri.xor %{{.*}}, 2 : (!x86.reg<>) -> !x86.reg<>
%ri_mov = x86.ri.mov %0, 2 : (!x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.ri.mov %{{.*}}, 2 : (!x86.reg<>) -> !x86.reg<>