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
