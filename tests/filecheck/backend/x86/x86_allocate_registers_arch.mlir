// RUN: xdsl-opt -p 'x86-allocate-registers{arch=avx512}' %s | filecheck %s
// RUN: xdsl-opt -p 'x86-allocate-registers' --verify-diagnostics %s | filecheck %s --check-prefix=CHECK-VEX

// zmm16-31 can only be named through EVEX, so they are allocatable on an AVX-512
// target and nowhere else. Seventeen simultaneously live vectors need one of them,
// which the default, VEX-only target cannot hand out.

// CHECK-LABEL:  @seventeen_live_vectors
// CHECK-NOT:  Out of registers.

// CHECK-VEX:  Out of registers.
x86_func.func @seventeen_live_vectors() {
  %ptr = "test.op"() : () -> !x86.reg64<rdi>
  %v0 = x86.dm.vmovapd [%ptr] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %v1 = x86.dm.vmovapd [%ptr + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %v2 = x86.dm.vmovapd [%ptr + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %v3 = x86.dm.vmovapd [%ptr + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %v4 = x86.dm.vmovapd [%ptr + 256] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %v5 = x86.dm.vmovapd [%ptr + 320] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %v6 = x86.dm.vmovapd [%ptr + 384] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %v7 = x86.dm.vmovapd [%ptr + 448] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %v8 = x86.dm.vmovapd [%ptr + 512] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %v9 = x86.dm.vmovapd [%ptr + 576] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %v10 = x86.dm.vmovapd [%ptr + 640] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %v11 = x86.dm.vmovapd [%ptr + 704] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %v12 = x86.dm.vmovapd [%ptr + 768] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %v13 = x86.dm.vmovapd [%ptr + 832] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %v14 = x86.dm.vmovapd [%ptr + 896] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %v15 = x86.dm.vmovapd [%ptr + 960] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %v16 = x86.dm.vmovapd [%ptr + 1024] : (!x86.reg64<rdi>) -> !x86.avx512reg
  x86.ms.vmovapd [%ptr], %v0 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86.ms.vmovapd [%ptr + 64], %v1 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86.ms.vmovapd [%ptr + 128], %v2 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86.ms.vmovapd [%ptr + 192], %v3 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86.ms.vmovapd [%ptr + 256], %v4 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86.ms.vmovapd [%ptr + 320], %v5 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86.ms.vmovapd [%ptr + 384], %v6 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86.ms.vmovapd [%ptr + 448], %v7 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86.ms.vmovapd [%ptr + 512], %v8 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86.ms.vmovapd [%ptr + 576], %v9 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86.ms.vmovapd [%ptr + 640], %v10 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86.ms.vmovapd [%ptr + 704], %v11 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86.ms.vmovapd [%ptr + 768], %v12 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86.ms.vmovapd [%ptr + 832], %v13 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86.ms.vmovapd [%ptr + 896], %v14 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86.ms.vmovapd [%ptr + 960], %v15 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86.ms.vmovapd [%ptr + 1024], %v16 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
  x86_func.ret
}
