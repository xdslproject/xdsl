// RUN: xdsl-opt -p dce %s | filecheck %s

// CHECK: %c42 = x86.di.mov 42 : () -> !x86.reg
%c42 = x86.di.mov 42 : () -> !x86.reg

// CHECK-NEXT: %unallocated = x86.ds.mov %c42 : (!x86.reg) -> !x86.reg
%unallocated = x86.ds.mov %c42 : (!x86.reg) -> !x86.reg
// CHECK-NEXT: %allocated = x86.ds.mov %c42 : (!x86.reg) -> !x86.reg<rax> 
%allocated = x86.ds.mov %c42 : (!x86.reg) -> !x86.reg<rax>
// CHECK-NEXT: %rsp = "test.op"() : () -> !x86.reg<rsp>
%rsp = "test.op"() : () -> !x86.reg<rsp>
// CHECK-NEXT: %rdx = "test.op"() : () -> !x86.reg<rdx>
%rdx = "test.op"() : () -> !x86.reg<rdx>

// Write effects don't get eliminated even if the result is unused

// CHECK-NEXT: x86.ms.add %unallocated, %unallocated, 0 : (!x86.reg, !x86.reg) -> ()
x86.ms.add %unallocated, %unallocated : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.ms.sub %unallocated, %unallocated, -8 : (!x86.reg, !x86.reg) -> ()
x86.ms.sub %unallocated, %unallocated, -8 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.ms.and %unallocated, %unallocated, 8 : (!x86.reg, !x86.reg) -> ()
x86.ms.and %unallocated, %unallocated, 8 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.ms.or %unallocated, %unallocated, 8 : (!x86.reg, !x86.reg) -> ()
x86.ms.or %unallocated, %unallocated, 8 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.ms.xor %unallocated, %unallocated, 8 : (!x86.reg, !x86.reg) -> ()
x86.ms.xor %unallocated, %unallocated, 8 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: x86.ms.mov %unallocated, %unallocated, 8 : (!x86.reg, !x86.reg) -> ()
x86.ms.mov %unallocated, %unallocated, 8 : (!x86.reg, !x86.reg) -> ()

// CHECK-NEXT: x86.mi.add %unallocated, 2 : (!x86.reg) -> ()
x86.mi.add %unallocated, 2 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.sub %unallocated, 2, -8 : (!x86.reg) -> ()
x86.mi.sub %unallocated, 2, -8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.and %unallocated, 2, 8 : (!x86.reg) -> ()
x86.mi.and %unallocated, 2, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.or %unallocated, 2, 8 : (!x86.reg) -> ()
x86.mi.or %unallocated, 2, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.xor %unallocated, 2, 8 : (!x86.reg) -> ()
x86.mi.xor %unallocated, 2, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.mi.mov %unallocated, 2, 8 : (!x86.reg) -> ()
x86.mi.mov %unallocated, 2, 8 : (!x86.reg) -> ()

// CHECK-NEXT: %m_push_rsp = x86.m.push %rsp, %unallocated, 0 : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
%m_push_rsp = x86.m.push %rsp, %unallocated : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>

// CHECK-NEXT: %m_pop_rsp = x86.m.pop %rsp, %unallocated, 8 : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>
%m_pop_rsp = x86.m.pop %rsp, %unallocated, 8 : (!x86.reg<rsp>, !x86.reg) -> !x86.reg<rsp>

// CHECK-NEXT: x86.m.neg %unallocated, 0 : (!x86.reg) -> ()
x86.m.neg %unallocated : (!x86.reg) -> ()
// CHECK-NEXT: x86.m.not %unallocated, 8 : (!x86.reg) -> ()
x86.m.not %unallocated, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.m.inc %unallocated, 8 : (!x86.reg) -> ()
x86.m.inc %unallocated, 8 : (!x86.reg) -> ()
// CHECK-NEXT: x86.m.dec %unallocated, 8 : (!x86.reg) -> ()
x86.m.dec %unallocated, 8 : (!x86.reg) -> ()
