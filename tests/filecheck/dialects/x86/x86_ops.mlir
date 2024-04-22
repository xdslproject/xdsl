// RUN: XDSL_ROUNDTRIP
%a = "test.op"() : () -> !x86.reg<rax>
// CHECK: %{{.*}} = "test.op"() : () -> !x86.reg<rax>

%0, %1 = "test.op"() : () -> (!x86.reg<>, !x86.reg<>)
%rsp = "test.op"() : () -> !x86.reg<rsp>

%add = x86.rr_add %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK: %{{.*}} = x86.rr_add %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%sub = x86.rr_sub %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rr_sub %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%mul = x86.rr_imul %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rr_imul %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%and = x86.rr_and %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rr_and %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%or = x86.rr_or %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rr_or %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%xor = x86.rr_xor %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rr_xor %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
%mov = x86.rr_mov %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.rr_mov %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
x86.r_push %0 : (!x86.reg<>) -> ()
// CHECK-NEXT: x86.r_push %{{.*}} : (!x86.reg<>)
%pop, %poprsp = x86.r_pop %rsp : (!x86.reg<rsp>) -> (!x86.reg<>, !x86.reg<rsp>)
// CHECK-NEXT: %{{.*}}, %{{.*}} = x86.r_pop %{{.*}} : (!x86.reg<rsp>) -> (!x86.reg<>, !x86.reg<rsp>)
%not = x86.r_not %0 : (!x86.reg<>) -> !x86.reg<>
// CHECK-NEXT: %{{.*}} = x86.r_not %{{.*}} : (!x86.reg<>) -> !x86.reg<>
