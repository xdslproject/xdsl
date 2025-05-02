// RUN: xdsl-opt -p x86-allocate-registers -t x86-asm %s | filecheck %s

// C: 4x4xf32 = A: 4x8xf32 * B: 8x4xf32
x86_func.func @matmul(%rdi: !x86.reg<rdi>, %rsi: !x86.reg<rsi>, %rcx: !x86.reg<rcx>) {
    // Load rows of A
    %a_row_0 = x86.rm.vmovups %rdi, 0 : (!x86.reg<rdi>) -> (!x86.avx2reg)
    %a_row_1 = x86.rm.vmovups %rdi, 32 : (!x86.reg<rdi>) -> (!x86.avx2reg)
    %a_row_2 = x86.rm.vmovups %rdi, 64 : (!x86.reg<rdi>) -> (!x86.avx2reg)
    %a_row_3 = x86.rm.vmovups %rdi, 96 : (!x86.reg<rdi>) -> (!x86.avx2reg)
    // Initialize the accumulators (rows of C)
    %c0_tmp0 = x86.rm.vmovups %rcx, 0 : (!x86.reg<rcx>) -> !x86.avx2reg
    %c1_tmp0 = x86.rm.vmovups %rcx, 32 : (!x86.reg<rcx>) -> !x86.avx2reg
    %c2_tmp0 = x86.rm.vmovups %rcx, 64 : (!x86.reg<rcx>) -> !x86.avx2reg
    %c3_tmp0 = x86.rm.vmovups %rcx, 96 : (!x86.reg<rcx>) -> !x86.avx2reg
    // Load column 0 of B
    %b_col_0 = x86.rm.vbroadcastss %rsi, 0 : (!x86.reg<rsi>) -> !x86.avx2reg
    // Reduction
    %c0_tmp1 = x86.rrr.vfmadd231ps %c0_tmp0, %b_col_0, %a_row_0 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
    %c1_tmp1 = x86.rrr.vfmadd231ps %c1_tmp0, %b_col_0, %a_row_1 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
    %c2_tmp1 = x86.rrr.vfmadd231ps %c2_tmp0, %b_col_0, %a_row_2 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
    %c3_tmp1 = x86.rrr.vfmadd231ps %c3_tmp0, %b_col_0, %a_row_3 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
    // Load column 1 of B
    %b_col_1 = x86.rm.vbroadcastss %rsi, 4 : (!x86.reg<rsi>) -> !x86.avx2reg
    // Reduction
    %c0_tmp2 = x86.rrr.vfmadd231ps %c0_tmp1, %b_col_1, %a_row_0 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
    %c1_tmp2 = x86.rrr.vfmadd231ps %c1_tmp1, %b_col_1, %a_row_1 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
    %c2_tmp2 = x86.rrr.vfmadd231ps %c2_tmp1, %b_col_1, %a_row_2 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
    %c3_tmp2 = x86.rrr.vfmadd231ps %c3_tmp1, %b_col_1, %a_row_3 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
    // Load column 2 of B
    %b_col_2 = x86.rm.vbroadcastss %rsi, 8 : (!x86.reg<rsi>) -> !x86.avx2reg
    // Reduction
    %c0_tmp3 = x86.rrr.vfmadd231ps %c0_tmp2, %b_col_2, %a_row_0 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
    %c1_tmp3 = x86.rrr.vfmadd231ps %c1_tmp2, %b_col_2, %a_row_1 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
    %c2_tmp3 = x86.rrr.vfmadd231ps %c2_tmp2, %b_col_2, %a_row_2 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
    %c3_tmp3 = x86.rrr.vfmadd231ps %c3_tmp2, %b_col_2, %a_row_3 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
    // Load column 3 of B
    %b_col_3 = x86.rm.vbroadcastss %rsi, 12 : (!x86.reg<rsi>) -> !x86.avx2reg
    // Reduction
    %c0 = x86.rrr.vfmadd231ps %c0_tmp3, %b_col_3, %a_row_0 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
    %c1 = x86.rrr.vfmadd231ps %c1_tmp3, %b_col_3, %a_row_1 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
    %c2 = x86.rrr.vfmadd231ps %c2_tmp3, %b_col_3, %a_row_2 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
    %c3 = x86.rrr.vfmadd231ps %c3_tmp3, %b_col_3, %a_row_3 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
   // Store the results (rows of C)
    x86.mr.vmovups %rcx, %c0, 0 : (!x86.reg<rcx>,!x86.avx2reg) -> ()
    x86.mr.vmovups %rcx, %c1, 32 : (!x86.reg<rcx>,!x86.avx2reg) -> ()
    x86.mr.vmovups %rcx,%c2,64 : (!x86.reg<rcx>,!x86.avx2reg) -> ()
    x86.mr.vmovups %rcx,%c3,96 : (!x86.reg<rcx>,!x86.avx2reg) -> ()

    x86_func.ret
}

// CHECK:       matmul:
// CHECK-NEXT:      vmovups ymm8, [rdi]
// CHECK-NEXT:      vmovups ymm7, [rdi+32]
// CHECK-NEXT:      vmovups ymm6, [rdi+64]
// CHECK-NEXT:      vmovups ymm5, [rdi+96]
// CHECK-NEXT:      vmovups ymm3, [rcx]
// CHECK-NEXT:      vmovups ymm2, [rcx+32]
// CHECK-NEXT:      vmovups ymm1, [rcx+64]
// CHECK-NEXT:      vmovups ymm0, [rcx+96]
// CHECK-NEXT:      vbroadcastss ymm4, [rsi]
// CHECK-NEXT:      vfmadd231ps ymm3, ymm4, ymm8
// CHECK-NEXT:      vfmadd231ps ymm2, ymm4, ymm7
// CHECK-NEXT:      vfmadd231ps ymm1, ymm4, ymm6
// CHECK-NEXT:      vfmadd231ps ymm0, ymm4, ymm5
// CHECK-NEXT:      vbroadcastss ymm4, [rsi+4]
// CHECK-NEXT:      vfmadd231ps ymm3, ymm4, ymm8
// CHECK-NEXT:      vfmadd231ps ymm2, ymm4, ymm7
// CHECK-NEXT:      vfmadd231ps ymm1, ymm4, ymm6
// CHECK-NEXT:      vfmadd231ps ymm0, ymm4, ymm5
// CHECK-NEXT:      vbroadcastss ymm4, [rsi+8]
// CHECK-NEXT:      vfmadd231ps ymm3, ymm4, ymm8
// CHECK-NEXT:      vfmadd231ps ymm2, ymm4, ymm7
// CHECK-NEXT:      vfmadd231ps ymm1, ymm4, ymm6
// CHECK-NEXT:      vfmadd231ps ymm0, ymm4, ymm5
// CHECK-NEXT:      vbroadcastss ymm4, [rsi+12]
// CHECK-NEXT:      vfmadd231ps ymm3, ymm4, ymm8
// CHECK-NEXT:      vfmadd231ps ymm2, ymm4, ymm7
// CHECK-NEXT:      vfmadd231ps ymm1, ymm4, ymm6
// CHECK-NEXT:      vfmadd231ps ymm0, ymm4, ymm5
// CHECK-NEXT:      vmovups [rcx], ymm3
// CHECK-NEXT:      vmovups [rcx+32], ymm2
// CHECK-NEXT:      vmovups [rcx+64], ymm1
// CHECK-NEXT:      vmovups [rcx+96], ymm0
// CHECK-NEXT:      ret
