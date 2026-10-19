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
// CHECK-NEXT: %zmm = "test.op"() : () -> !x86.avx512reg
%zmm = "test.op"() : () -> !x86.avx512reg
// CHECK-NEXT: %xmm = "test.op"() : () -> !x86.ssereg
%xmm = "test.op"() : () -> !x86.ssereg

// Unused reads can be eliminated

// CHECK-NOT: x86.dsm.vmulpd
%vmulpd = x86.dsm.vmulpd %zmm, [%unallocated] : (!x86.avx512reg, !x86.reg64) -> !x86.avx512reg
// CHECK-NOT: x86.dsm.vaddsd
%vaddsd = x86.dsm.vaddsd %xmm, [%unallocated] : (!x86.ssereg, !x86.reg64) -> !x86.ssereg

// Write effects don't get eliminated even if the result is unused

// CHECK-NEXT: x86.ms.vmovsd [%unallocated], %xmm : (!x86.reg64, !x86.ssereg) -> ()
x86.ms.vmovsd [%unallocated], %xmm : (!x86.reg64, !x86.ssereg) -> ()
// CHECK-NEXT: x86.ms.add [%unallocated], %unallocated : (!x86.reg64, !x86.reg64) -> ()
x86.ms.add [%unallocated], %unallocated : (!x86.reg64, !x86.reg64) -> ()
// CHECK-NEXT: x86.ms.sub [%unallocated + -8], %unallocated : (!x86.reg64, !x86.reg64) -> ()
x86.ms.sub [%unallocated + -8], %unallocated : (!x86.reg64, !x86.reg64) -> ()
// CHECK-NEXT: x86.ms.and [%unallocated + 8], %unallocated : (!x86.reg64, !x86.reg64) -> ()
x86.ms.and [%unallocated + 8], %unallocated : (!x86.reg64, !x86.reg64) -> ()
// CHECK-NEXT: x86.ms.or [%unallocated + 8], %unallocated : (!x86.reg64, !x86.reg64) -> ()
x86.ms.or [%unallocated + 8], %unallocated : (!x86.reg64, !x86.reg64) -> ()
// CHECK-NEXT: x86.ms.xor [%unallocated + 8], %unallocated : (!x86.reg64, !x86.reg64) -> ()
x86.ms.xor [%unallocated + 8], %unallocated : (!x86.reg64, !x86.reg64) -> ()
// CHECK-NEXT: x86.ms.mov [%unallocated + 8], %unallocated : (!x86.reg64, !x86.reg64) -> ()
x86.ms.mov [%unallocated + 8], %unallocated : (!x86.reg64, !x86.reg64) -> ()

// CHECK-NEXT: x86.mi.add [%unallocated], 2 : (!x86.reg64) -> ()
x86.mi.add [%unallocated], 2 : (!x86.reg64) -> ()
// CHECK-NEXT: x86.mi.sub [%unallocated + -8], 2 : (!x86.reg64) -> ()
x86.mi.sub [%unallocated + -8], 2 : (!x86.reg64) -> ()
// CHECK-NEXT: x86.mi.and [%unallocated + 8], 2 : (!x86.reg64) -> ()
x86.mi.and [%unallocated + 8], 2 : (!x86.reg64) -> ()
// CHECK-NEXT: x86.mi.or [%unallocated + 8], 2 : (!x86.reg64) -> ()
x86.mi.or [%unallocated + 8], 2 : (!x86.reg64) -> ()
// CHECK-NEXT: x86.mi.xor [%unallocated + 8], 2 : (!x86.reg64) -> ()
x86.mi.xor [%unallocated + 8], 2 : (!x86.reg64) -> ()
// CHECK-NEXT: x86.mi.mov [%unallocated + 8], 2 : (!x86.reg64) -> ()
x86.mi.mov [%unallocated + 8], 2 : (!x86.reg64) -> ()

// CHECK-NEXT: %m_push_rsp = x86.m.push %rsp, [%unallocated] : (!x86.reg64<rsp>, !x86.reg64) -> !x86.reg64<rsp>
%m_push_rsp = x86.m.push %rsp, [%unallocated] : (!x86.reg64<rsp>, !x86.reg64) -> !x86.reg64<rsp>

// CHECK-NEXT: %m_pop_rsp = x86.m.pop %rsp, [%unallocated + 8] : (!x86.reg64<rsp>, !x86.reg64) -> !x86.reg64<rsp>
%m_pop_rsp = x86.m.pop %rsp, [%unallocated + 8] : (!x86.reg64<rsp>, !x86.reg64) -> !x86.reg64<rsp>

// CHECK-NEXT: x86.m.neg [%unallocated] : (!x86.reg64) -> ()
x86.m.neg [%unallocated] : (!x86.reg64) -> ()
// CHECK-NEXT: x86.m.not [%unallocated + 8] : (!x86.reg64) -> ()
x86.m.not [%unallocated + 8] : (!x86.reg64) -> ()
// CHECK-NEXT: x86.m.inc [%unallocated + 8] : (!x86.reg64) -> ()
x86.m.inc [%unallocated + 8] : (!x86.reg64) -> ()
// CHECK-NEXT: x86.m.dec [%unallocated + 8] : (!x86.reg64) -> ()
x86.m.dec [%unallocated + 8] : (!x86.reg64) -> ()
