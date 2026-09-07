// RUN: xdsl-opt --verify-diagnostics --split-input-file %s | filecheck %s

x86_func.func @funcyasm() {
    %3 = x86.get_register : !x86.reg64<rax>
    %4 = x86.get_register : !x86.reg64<rdx>
    %rflags = x86.ss.cmp %3, %4 : (!x86.reg64<rax>, !x86.reg64<rdx>) -> !x86.rflags<rflags>
    x86.fallthrough ^fallthrough()
    // CHECK: Operation does not verify: Fallthrough op successor must immediately follow its parent.
    ^other:
    x86.label "other"
    ^fallthrough:
    x86.label "fallthrough"
    x86_func.ret
}

// -----

// Mismatched input and output counts
%0, %1, %2 = "test.op"() : () -> (!x86.reg64<rax>, !x86.reg64<rbx>, !x86.reg64<rcx>)
x86.parallel_mov %0, %1, %2 : (!x86.reg64<rax>, !x86.reg64<rbx>, !x86.reg64<rcx>) -> (!x86.reg64<rdx>, !x86.reg64<rsi>)

// CHECK: %0, %1, %2 = "test.op"() : () -> (!x86.reg64<rax>, !x86.reg64<rbx>, !x86.reg64<rcx>)
// CHECK: Input count must match output count. Num inputs: 3, Num outputs: 2

// -----

// Moving from int register to vector register
%0, %1 = "test.op"() : () -> (!x86.reg64<r8>, !x86.reg64<r9>)
x86.parallel_mov %0, %1 : (!x86.reg64<r8>, !x86.reg64<r9>) -> (!x86.ssereg, !x86.ssereg)

// CHECK: %0, %1 = "test.op"() : () -> (!x86.reg64<r8>, !x86.reg64<r9>)
// CHECK: Input type must match output type.

// -----

// Duplicated output registers
%0, %1 = "test.op"() : () -> (!x86.reg64<r10>, !x86.reg64<r11>)
x86.parallel_mov %0, %1 : (!x86.reg64<r10>, !x86.reg64<r11>) -> (!x86.reg64<r12>, !x86.reg64<r12>)

// CHECK: %0, %1 = "test.op"() : () -> (!x86.reg64<r10>, !x86.reg64<r11>)
// CHECK: Outputs must be unallocated or distinct.

// -----

// vmulpd memory form destination and vector source must have matching widths.
%base, %zmm = "test.op"() : () -> (!x86.reg64, !x86.avx512reg)
%result = x86.dsm.vmulpd %zmm, [%base] : (!x86.avx512reg, !x86.reg64) -> !x86.avx2reg

// CHECK: Expected all vector registers to have the same width

// -----

// vextractf64x4 requires a ZMM source and YMM destination.
%ymm = "test.op"() : () -> !x86.avx2reg
%result = x86.dsi.vextractf64x4 %ymm, 1 : (!x86.avx2reg) -> !x86.avx2reg

// CHECK: !x86.avx2reg should be of base attribute x86.avx512reg

// -----

// vextractf128 requires a YMM source and XMM destination.
%zmm = "test.op"() : () -> !x86.avx512reg
%result = x86.dsi.vextractf128 %zmm, 1 : (!x86.avx512reg) -> !x86.ssereg

// CHECK: !x86.avx512reg should be of base attribute x86.avx2reg

// -----

// Scalar memory-source vaddsd only accepts an XMM vector source and destination.
%base, %ymm = "test.op"() : () -> (!x86.reg64, !x86.avx2reg)
%result = x86.dsm.vaddsd %ymm, [%base] : (!x86.avx2reg, !x86.reg64) -> !x86.ssereg

// CHECK: !x86.avx2reg should be of base attribute x86.ssereg
// -----

// Extraction immediates must be ui8 attributes.
%ymm = "test.op"() : () -> !x86.avx2reg
%result = "x86.dsi.vextractf128"(%ymm) {immediate = 1 : i32} : (!x86.avx2reg) -> !x86.ssereg

// CHECK: Expected attribute ui8 but got i32
