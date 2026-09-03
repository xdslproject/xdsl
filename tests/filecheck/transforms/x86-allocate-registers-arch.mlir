// RUN: xdsl-opt -p 'x86-set-arch{arch=avx512},x86-allocate-registers' %s | filecheck %s
// RUN: xdsl-opt -p 'x86-allocate-registers' %s --verify-diagnostics | filecheck %s --check-prefix=VEX
// RUN: xdsl-opt -p 'x86-set-arch{arch=avx2},x86-allocate-registers' %s --verify-diagnostics | filecheck %s --check-prefix=VEX

// Seventeen values live at once. VEX names ymm0-15 only, so on a VEX target the
// seventeenth has nowhere to go. Before the target was recorded the allocator
// handed out ymm16 here regardless, which encodes as EVEX and is not decodable
// on a machine without AVX-512.

builtin.module {
  x86_func.func @seventeen_live(%base : !x86.reg64<rdi>) {
    %v0 = x86.dm.vmovupd [%base + 0] : (!x86.reg64<rdi>) -> !x86.avx2reg
    %v1 = x86.dm.vmovupd [%base + 32] : (!x86.reg64<rdi>) -> !x86.avx2reg
    %v2 = x86.dm.vmovupd [%base + 64] : (!x86.reg64<rdi>) -> !x86.avx2reg
    %v3 = x86.dm.vmovupd [%base + 96] : (!x86.reg64<rdi>) -> !x86.avx2reg
    %v4 = x86.dm.vmovupd [%base + 128] : (!x86.reg64<rdi>) -> !x86.avx2reg
    %v5 = x86.dm.vmovupd [%base + 160] : (!x86.reg64<rdi>) -> !x86.avx2reg
    %v6 = x86.dm.vmovupd [%base + 192] : (!x86.reg64<rdi>) -> !x86.avx2reg
    %v7 = x86.dm.vmovupd [%base + 224] : (!x86.reg64<rdi>) -> !x86.avx2reg
    %v8 = x86.dm.vmovupd [%base + 256] : (!x86.reg64<rdi>) -> !x86.avx2reg
    %v9 = x86.dm.vmovupd [%base + 288] : (!x86.reg64<rdi>) -> !x86.avx2reg
    %v10 = x86.dm.vmovupd [%base + 320] : (!x86.reg64<rdi>) -> !x86.avx2reg
    %v11 = x86.dm.vmovupd [%base + 352] : (!x86.reg64<rdi>) -> !x86.avx2reg
    %v12 = x86.dm.vmovupd [%base + 384] : (!x86.reg64<rdi>) -> !x86.avx2reg
    %v13 = x86.dm.vmovupd [%base + 416] : (!x86.reg64<rdi>) -> !x86.avx2reg
    %v14 = x86.dm.vmovupd [%base + 448] : (!x86.reg64<rdi>) -> !x86.avx2reg
    %v15 = x86.dm.vmovupd [%base + 480] : (!x86.reg64<rdi>) -> !x86.avx2reg
    %v16 = x86.dm.vmovupd [%base + 512] : (!x86.reg64<rdi>) -> !x86.avx2reg
    x86.ms.vmovapd [%base + 544], %v0 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86.ms.vmovapd [%base + 576], %v1 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86.ms.vmovapd [%base + 608], %v2 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86.ms.vmovapd [%base + 640], %v3 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86.ms.vmovapd [%base + 672], %v4 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86.ms.vmovapd [%base + 704], %v5 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86.ms.vmovapd [%base + 736], %v6 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86.ms.vmovapd [%base + 768], %v7 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86.ms.vmovapd [%base + 800], %v8 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86.ms.vmovapd [%base + 832], %v9 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86.ms.vmovapd [%base + 864], %v10 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86.ms.vmovapd [%base + 896], %v11 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86.ms.vmovapd [%base + 928], %v12 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86.ms.vmovapd [%base + 960], %v13 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86.ms.vmovapd [%base + 992], %v14 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86.ms.vmovapd [%base + 1024], %v15 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86.ms.vmovapd [%base + 1056], %v16 : (!x86.reg64<rdi>, !x86.avx2reg) -> ()
    x86_func.ret
  }
}

// AVX-512 can name the upper half of the bank, so all seventeen are allocated.
// CHECK:      builtin.module attributes {x86.arch = "avx512"}
// CHECK:      %v16 = x86.dm.vmovupd [%base + 512] : (!x86.reg64<rdi>) -> !x86.avx2reg<ymm{{[0-9]+}}>
// CHECK:      x86_func.ret

// VEX:        Out of registers.
// VEX:        Error allocating op
