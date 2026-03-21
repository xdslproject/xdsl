// RUN: xdsl-opt -p dce %s | filecheck %s

// CHECK: %c42 = x86.di.mov 42 : () -> !x86.reg64
%c42 = x86.di.mov 42 : () -> !x86.reg64

// CHECK-NEXT: %unallocated = x86.ds.mov %c42 : (!x86.reg64) -> !x86.reg64
%unallocated = x86.ds.mov %c42 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT: %allocated = x86.ds.mov %c42 : (!x86.reg64) -> !x86.reg64<rax>
%allocated = x86.ds.mov %c42 : (!x86.reg64) -> !x86.reg64<rax>
// CHECK-NEXT: %rsp = "test.op"() : () -> !x86.reg64<rsp>
%rsp = "test.op"() : () -> !x86.reg64<rsp>
// CHECK-NEXT: %rdx = "test.op"() : () -> !x86.reg64<rdx>
%rdx = "test.op"() : () -> !x86.reg64<rdx>

// Write effects don't get eliminated even if the result is unused

// CHECK-NEXT: x86.ms.add %unallocated, %unallocated : (!x86.reg64, !x86.reg64) -> ()
x86.ms.add %unallocated, %unallocated : (!x86.reg64, !x86.reg64) -> ()
// CHECK-NEXT: x86.ms.sub %unallocated, %unallocated, -8 : (!x86.reg64, !x86.reg64) -> ()
x86.ms.sub %unallocated, %unallocated, -8 : (!x86.reg64, !x86.reg64) -> ()
// CHECK-NEXT: x86.ms.and %unallocated, %unallocated, 8 : (!x86.reg64, !x86.reg64) -> ()
x86.ms.and %unallocated, %unallocated, 8 : (!x86.reg64, !x86.reg64) -> ()
// CHECK-NEXT: x86.ms.or %unallocated, %unallocated, 8 : (!x86.reg64, !x86.reg64) -> ()
x86.ms.or %unallocated, %unallocated, 8 : (!x86.reg64, !x86.reg64) -> ()
// CHECK-NEXT: x86.ms.xor %unallocated, %unallocated, 8 : (!x86.reg64, !x86.reg64) -> ()
x86.ms.xor %unallocated, %unallocated, 8 : (!x86.reg64, !x86.reg64) -> ()
// CHECK-NEXT: x86.ms.mov %unallocated, %unallocated, 8 : (!x86.reg64, !x86.reg64) -> ()
x86.ms.mov %unallocated, %unallocated, 8 : (!x86.reg64, !x86.reg64) -> ()

// CHECK-NEXT: x86.mi.add %unallocated, 2 : (!x86.reg64) -> ()
x86.mi.add %unallocated, 2 : (!x86.reg64) -> ()
// CHECK-NEXT: x86.mi.sub %unallocated, 2, -8 : (!x86.reg64) -> ()
x86.mi.sub %unallocated, 2, -8 : (!x86.reg64) -> ()
// CHECK-NEXT: x86.mi.and %unallocated, 2, 8 : (!x86.reg64) -> ()
x86.mi.and %unallocated, 2, 8 : (!x86.reg64) -> ()
// CHECK-NEXT: x86.mi.or %unallocated, 2, 8 : (!x86.reg64) -> ()
x86.mi.or %unallocated, 2, 8 : (!x86.reg64) -> ()
// CHECK-NEXT: x86.mi.xor %unallocated, 2, 8 : (!x86.reg64) -> ()
x86.mi.xor %unallocated, 2, 8 : (!x86.reg64) -> ()
// CHECK-NEXT: x86.mi.mov %unallocated, 2, 8 : (!x86.reg64) -> ()
x86.mi.mov %unallocated, 2, 8 : (!x86.reg64) -> ()

// CHECK-NEXT: %m_push_rsp = x86.m.push %rsp, %unallocated : (!x86.reg64<rsp>, !x86.reg64) -> !x86.reg64<rsp>
%m_push_rsp = x86.m.push %rsp, %unallocated : (!x86.reg64<rsp>, !x86.reg64) -> !x86.reg64<rsp>

// CHECK-NEXT: %m_pop_rsp = x86.m.pop %rsp, %unallocated, 8 : (!x86.reg64<rsp>, !x86.reg64) -> !x86.reg64<rsp>
%m_pop_rsp = x86.m.pop %rsp, %unallocated, 8 : (!x86.reg64<rsp>, !x86.reg64) -> !x86.reg64<rsp>

// CHECK-NEXT: x86.m.neg %unallocated : (!x86.reg64) -> ()
x86.m.neg %unallocated : (!x86.reg64) -> ()
// CHECK-NEXT: x86.m.not %unallocated, 8 : (!x86.reg64) -> ()
x86.m.not %unallocated, 8 : (!x86.reg64) -> ()
// CHECK-NEXT: x86.m.inc %unallocated, 8 : (!x86.reg64) -> ()
x86.m.inc %unallocated, 8 : (!x86.reg64) -> ()
// CHECK-NEXT: x86.m.dec %unallocated, 8 : (!x86.reg64) -> ()
x86.m.dec %unallocated, 8 : (!x86.reg64) -> ()
