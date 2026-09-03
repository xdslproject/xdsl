// RUN: xdsl-opt -p 'x86-allocate-registers{arch=avx512}' %s | filecheck %s
// RUN: xdsl-opt -p 'x86-allocate-registers' --verify-diagnostics %s | filecheck %s --check-prefix=CHECK-VEX

// zmm16-31 can only be named through EVEX, so they are allocatable on an AVX-512
// target and nowhere else. Seventeen simultaneously live vectors need one of them,
// which the default target cannot hand out.

// CHECK-VEX: Out of registers.

// CHECK-LABEL:  @seventeen_live_vectors
x86_func.func @seventeen_live_vectors() {

// CHECK-NEXT:      %ptr = "test.op"() : () -> !x86.reg64<rdi>
  %ptr = "test.op"() : () -> !x86.reg64<rdi>

// The seventeenth live vector is the one that has to come out of the EVEX-only half.
// CHECK-NEXT:      %v0 = x86.dm.vmovapd [%ptr] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm16>
  %v0 = x86.dm.vmovapd [%ptr] : (!x86.reg64<rdi>) -> !x86.avx512reg
// CHECK-NEXT:      %v1 = x86.dm.vmovapd [%ptr + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm15>
  %v1 = x86.dm.vmovapd [%ptr + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg
// CHECK-NEXT:      %v2 = x86.dm.vmovapd [%ptr + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm14>
  %v2 = x86.dm.vmovapd [%ptr + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg
// CHECK-NEXT:      %v3 = x86.dm.vmovapd [%ptr + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm13>
  %v3 = x86.dm.vmovapd [%ptr + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg
// CHECK-NEXT:      %v4 = x86.dm.vmovapd [%ptr + 256] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm12>
  %v4 = x86.dm.vmovapd [%ptr + 256] : (!x86.reg64<rdi>) -> !x86.avx512reg
// CHECK-NEXT:      %v5 = x86.dm.vmovapd [%ptr + 320] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm11>
  %v5 = x86.dm.vmovapd [%ptr + 320] : (!x86.reg64<rdi>) -> !x86.avx512reg
// CHECK-NEXT:      %v6 = x86.dm.vmovapd [%ptr + 384] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm10>
  %v6 = x86.dm.vmovapd [%ptr + 384] : (!x86.reg64<rdi>) -> !x86.avx512reg
// CHECK-NEXT:      %v7 = x86.dm.vmovapd [%ptr + 448] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm9>
  %v7 = x86.dm.vmovapd [%ptr + 448] : (!x86.reg64<rdi>) -> !x86.avx512reg
// CHECK-NEXT:      %v8 = x86.dm.vmovapd [%ptr + 512] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm8>
  %v8 = x86.dm.vmovapd [%ptr + 512] : (!x86.reg64<rdi>) -> !x86.avx512reg
// CHECK-NEXT:      %v9 = x86.dm.vmovapd [%ptr + 576] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm7>
  %v9 = x86.dm.vmovapd [%ptr + 576] : (!x86.reg64<rdi>) -> !x86.avx512reg
// CHECK-NEXT:      %v10 = x86.dm.vmovapd [%ptr + 640] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm6>
  %v10 = x86.dm.vmovapd [%ptr + 640] : (!x86.reg64<rdi>) -> !x86.avx512reg
// CHECK-NEXT:      %v11 = x86.dm.vmovapd [%ptr + 704] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm5>
  %v11 = x86.dm.vmovapd [%ptr + 704] : (!x86.reg64<rdi>) -> !x86.avx512reg
// CHECK-NEXT:      %v12 = x86.dm.vmovapd [%ptr + 768] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
  %v12 = x86.dm.vmovapd [%ptr + 768] : (!x86.reg64<rdi>) -> !x86.avx512reg
// CHECK-NEXT:      %v13 = x86.dm.vmovapd [%ptr + 832] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
  %v13 = x86.dm.vmovapd [%ptr + 832] : (!x86.reg64<rdi>) -> !x86.avx512reg
// CHECK-NEXT:      %v14 = x86.dm.vmovapd [%ptr + 896] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
  %v14 = x86.dm.vmovapd [%ptr + 896] : (!x86.reg64<rdi>) -> !x86.avx512reg
// CHECK-NEXT:      %v15 = x86.dm.vmovapd [%ptr + 960] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
  %v15 = x86.dm.vmovapd [%ptr + 960] : (!x86.reg64<rdi>) -> !x86.avx512reg
// CHECK-NEXT:      %v16 = x86.dm.vmovapd [%ptr + 1024] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm0>
  %v16 = x86.dm.vmovapd [%ptr + 1024] : (!x86.reg64<rdi>) -> !x86.avx512reg

// CHECK-NEXT:      x86.ms.vmovapd [%ptr], %v0 : (!x86.reg64<rdi>, !x86.avx512reg<zmm16>) -> ()
  x86.ms.vmovapd [%ptr], %v0 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%ptr + 64], %v1 : (!x86.reg64<rdi>, !x86.avx512reg<zmm15>) -> ()
  x86.ms.vmovapd [%ptr + 64], %v1 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%ptr + 128], %v2 : (!x86.reg64<rdi>, !x86.avx512reg<zmm14>) -> ()
  x86.ms.vmovapd [%ptr + 128], %v2 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%ptr + 192], %v3 : (!x86.reg64<rdi>, !x86.avx512reg<zmm13>) -> ()
  x86.ms.vmovapd [%ptr + 192], %v3 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%ptr + 256], %v4 : (!x86.reg64<rdi>, !x86.avx512reg<zmm12>) -> ()
  x86.ms.vmovapd [%ptr + 256], %v4 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%ptr + 320], %v5 : (!x86.reg64<rdi>, !x86.avx512reg<zmm11>) -> ()
  x86.ms.vmovapd [%ptr + 320], %v5 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%ptr + 384], %v6 : (!x86.reg64<rdi>, !x86.avx512reg<zmm10>) -> ()
  x86.ms.vmovapd [%ptr + 384], %v6 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%ptr + 448], %v7 : (!x86.reg64<rdi>, !x86.avx512reg<zmm9>) -> ()
  x86.ms.vmovapd [%ptr + 448], %v7 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%ptr + 512], %v8 : (!x86.reg64<rdi>, !x86.avx512reg<zmm8>) -> ()
  x86.ms.vmovapd [%ptr + 512], %v8 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%ptr + 576], %v9 : (!x86.reg64<rdi>, !x86.avx512reg<zmm7>) -> ()
  x86.ms.vmovapd [%ptr + 576], %v9 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%ptr + 640], %v10 : (!x86.reg64<rdi>, !x86.avx512reg<zmm6>) -> ()
  x86.ms.vmovapd [%ptr + 640], %v10 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%ptr + 704], %v11 : (!x86.reg64<rdi>, !x86.avx512reg<zmm5>) -> ()
  x86.ms.vmovapd [%ptr + 704], %v11 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%ptr + 768], %v12 : (!x86.reg64<rdi>, !x86.avx512reg<zmm4>) -> ()
  x86.ms.vmovapd [%ptr + 768], %v12 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%ptr + 832], %v13 : (!x86.reg64<rdi>, !x86.avx512reg<zmm3>) -> ()
  x86.ms.vmovapd [%ptr + 832], %v13 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%ptr + 896], %v14 : (!x86.reg64<rdi>, !x86.avx512reg<zmm2>) -> ()
  x86.ms.vmovapd [%ptr + 896], %v14 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%ptr + 960], %v15 : (!x86.reg64<rdi>, !x86.avx512reg<zmm1>) -> ()
  x86.ms.vmovapd [%ptr + 960], %v15 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%ptr + 1024], %v16 : (!x86.reg64<rdi>, !x86.avx512reg<zmm0>) -> ()
  x86.ms.vmovapd [%ptr + 1024], %v16 : (!x86.reg64<rdi>, !x86.avx512reg) -> ()

// CHECK-NEXT:      x86_func.ret
  x86_func.ret
}
