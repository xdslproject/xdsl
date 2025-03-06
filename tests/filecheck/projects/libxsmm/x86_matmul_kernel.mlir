// RUN: xdsl-opt -t x86-asm %s | filecheck %s

// C: 4x4xf32 = A: 4x8xf32 * B: 8x4xf32
x86_func.func @matmul(%rdi: !x86.reg<rdi>, %rsi: !x86.reg<rsi>, %rcx: !x86.reg<rcx>) {
    // Load rows of A
    %a_row_0 = x86.rm.vmovups %rdi, 0 : (!x86.reg<rdi>) -> (!x86.avx2reg<ymm0>)
    %a_row_1 = x86.rm.vmovups %rdi, 32 : (!x86.reg<rdi>) -> (!x86.avx2reg<ymm1>)
    %a_row_2 = x86.rm.vmovups %rdi, 64 : (!x86.reg<rdi>) -> (!x86.avx2reg<ymm2>)
    %a_row_3 = x86.rm.vmovups %rdi, 96 : (!x86.reg<rdi>) -> (!x86.avx2reg<ymm3>)
    // Initialize the accumulators (rows of C)
    %c0_tmp0 = x86.rm.vmovups %rcx, 0 : (!x86.reg<rcx>) -> !x86.avx2reg<ymm4>
    %c1_tmp0 = x86.rm.vmovups %rcx, 32 : (!x86.reg<rcx>) -> !x86.avx2reg<ymm5>
    %c2_tmp0 = x86.rm.vmovups %rcx, 64 : (!x86.reg<rcx>) -> !x86.avx2reg<ymm6>
    %c3_tmp0 = x86.rm.vmovups %rcx, 96 : (!x86.reg<rcx>) -> !x86.avx2reg<ymm7>
    // Load column 0 of B
    %b_col_0 = x86.rm.vbroadcastss %rsi, 0 : (!x86.reg<rsi>) -> !x86.avx2reg<ymm8>
    // Reduction
    %c0_tmp1 = x86.rrr.vfmadd231ps %c0_tmp0, %b_col_0, %a_row_0 : (!x86.avx2reg<ymm4>, !x86.avx2reg<ymm8>, !x86.avx2reg<ymm0>) -> !x86.avx2reg<ymm4>
    %c1_tmp1 = x86.rrr.vfmadd231ps %c1_tmp0, %b_col_0, %a_row_1 : (!x86.avx2reg<ymm5>, !x86.avx2reg<ymm8>, !x86.avx2reg<ymm1>) -> !x86.avx2reg<ymm5>
    %c2_tmp1 = x86.rrr.vfmadd231ps %c2_tmp0, %b_col_0, %a_row_2 : (!x86.avx2reg<ymm6>, !x86.avx2reg<ymm8>, !x86.avx2reg<ymm2>) -> !x86.avx2reg<ymm6>
    %c3_tmp1 = x86.rrr.vfmadd231ps %c3_tmp0, %b_col_0, %a_row_3 : (!x86.avx2reg<ymm7>, !x86.avx2reg<ymm8>, !x86.avx2reg<ymm3>) -> !x86.avx2reg<ymm7>
    // Load column 1 of B
    %b_col_1 = x86.rm.vbroadcastss %rsi, 4 : (!x86.reg<rsi>) -> !x86.avx2reg<ymm9>
    // Reduction
    %c0_tmp2 = x86.rrr.vfmadd231ps %c0_tmp1, %b_col_1, %a_row_0 : (!x86.avx2reg<ymm4>, !x86.avx2reg<ymm9>, !x86.avx2reg<ymm0>) -> !x86.avx2reg<ymm4>
    %c1_tmp2 = x86.rrr.vfmadd231ps %c1_tmp1, %b_col_1, %a_row_1 : (!x86.avx2reg<ymm5>, !x86.avx2reg<ymm9>, !x86.avx2reg<ymm1>) -> !x86.avx2reg<ymm5>
    %c2_tmp2 = x86.rrr.vfmadd231ps %c2_tmp1, %b_col_1, %a_row_2 : (!x86.avx2reg<ymm6>, !x86.avx2reg<ymm9>, !x86.avx2reg<ymm2>) -> !x86.avx2reg<ymm6>
    %c3_tmp2 = x86.rrr.vfmadd231ps %c3_tmp1, %b_col_1, %a_row_3 : (!x86.avx2reg<ymm7>, !x86.avx2reg<ymm9>, !x86.avx2reg<ymm3>) -> !x86.avx2reg<ymm7>
    // Load column 2 of B
    %b_col_2 = x86.rm.vbroadcastss %rsi, 8 : (!x86.reg<rsi>) -> !x86.avx2reg<ymm10>
    // Reduction
    %c0_tmp3 = x86.rrr.vfmadd231ps %c0_tmp2, %b_col_2, %a_row_0 : (!x86.avx2reg<ymm4>, !x86.avx2reg<ymm10>, !x86.avx2reg<ymm0>) -> !x86.avx2reg<ymm4>
    %c1_tmp3 = x86.rrr.vfmadd231ps %c1_tmp2, %b_col_2, %a_row_1 : (!x86.avx2reg<ymm5>, !x86.avx2reg<ymm10>, !x86.avx2reg<ymm1>) -> !x86.avx2reg<ymm5>
    %c2_tmp3 = x86.rrr.vfmadd231ps %c2_tmp2, %b_col_2, %a_row_2 : (!x86.avx2reg<ymm6>, !x86.avx2reg<ymm10>, !x86.avx2reg<ymm2>) -> !x86.avx2reg<ymm6>
    %c3_tmp3 = x86.rrr.vfmadd231ps %c3_tmp2, %b_col_2, %a_row_3 : (!x86.avx2reg<ymm7>, !x86.avx2reg<ymm10>, !x86.avx2reg<ymm3>) -> !x86.avx2reg<ymm7>
    // Load column 3 of B
    %b_col_3 = x86.rm.vbroadcastss %rsi, 12 : (!x86.reg<rsi>) -> !x86.avx2reg<ymm11>
    // Reduction
    %c0 = x86.rrr.vfmadd231ps %c0_tmp3, %b_col_3, %a_row_0 : (!x86.avx2reg<ymm4>, !x86.avx2reg<ymm11>, !x86.avx2reg<ymm0>) -> !x86.avx2reg<ymm4>
    %c1 = x86.rrr.vfmadd231ps %c1_tmp3, %b_col_3, %a_row_1 : (!x86.avx2reg<ymm5>, !x86.avx2reg<ymm11>, !x86.avx2reg<ymm1>) -> !x86.avx2reg<ymm5>
    %c2 = x86.rrr.vfmadd231ps %c2_tmp3, %b_col_3, %a_row_2 : (!x86.avx2reg<ymm6>, !x86.avx2reg<ymm11>, !x86.avx2reg<ymm2>) -> !x86.avx2reg<ymm6>
    %c3 = x86.rrr.vfmadd231ps %c3_tmp3, %b_col_3, %a_row_3 : (!x86.avx2reg<ymm7>, !x86.avx2reg<ymm11>, !x86.avx2reg<ymm3>) -> !x86.avx2reg<ymm7>
   // Store the results (rows of C)
    x86.mr.vmovups %rcx, %c0, 0 : (!x86.reg<rcx>,!x86.avx2reg<ymm4>) -> ()
    x86.mr.vmovups %rcx, %c1, 32 : (!x86.reg<rcx>,!x86.avx2reg<ymm5>) -> ()
    x86.mr.vmovups %rcx,%c2,64 : (!x86.reg<rcx>,!x86.avx2reg<ymm6>) -> ()
    x86.mr.vmovups %rcx,%c3,96 : (!x86.reg<rcx>,!x86.avx2reg<ymm7>) -> ()
    
    x86_func.ret
}

// CHECK:       matmul:
// CHECK-NEXT:      vmovups ymm0, [rdi]
// CHECK-NEXT:      vmovups ymm1, [rdi+32]
// CHECK-NEXT:      vmovups ymm2, [rdi+64]
// CHECK-NEXT:      vmovups ymm3, [rdi+96]
// CHECK-NEXT:      vmovups ymm4, [rcx]
// CHECK-NEXT:      vmovups ymm5, [rcx+32]
// CHECK-NEXT:      vmovups ymm6, [rcx+64]
// CHECK-NEXT:      vmovups ymm7, [rcx+96]
// CHECK-NEXT:      vbroadcastss ymm8, [rsi]
// CHECK-NEXT:      vfmadd231ps ymm4, ymm8, ymm0
// CHECK-NEXT:      vfmadd231ps ymm5, ymm8, ymm1
// CHECK-NEXT:      vfmadd231ps ymm6, ymm8, ymm2
// CHECK-NEXT:      vfmadd231ps ymm7, ymm8, ymm3
// CHECK-NEXT:      vbroadcastss ymm9, [rsi+4]
// CHECK-NEXT:      vfmadd231ps ymm4, ymm9, ymm0
// CHECK-NEXT:      vfmadd231ps ymm5, ymm9, ymm1
// CHECK-NEXT:      vfmadd231ps ymm6, ymm9, ymm2
// CHECK-NEXT:      vfmadd231ps ymm7, ymm9, ymm3
// CHECK-NEXT:      vbroadcastss ymm10, [rsi+8]
// CHECK-NEXT:      vfmadd231ps ymm4, ymm10, ymm0
// CHECK-NEXT:      vfmadd231ps ymm5, ymm10, ymm1
// CHECK-NEXT:      vfmadd231ps ymm6, ymm10, ymm2
// CHECK-NEXT:      vfmadd231ps ymm7, ymm10, ymm3
// CHECK-NEXT:      vbroadcastss ymm11, [rsi+12]
// CHECK-NEXT:      vfmadd231ps ymm4, ymm11, ymm0
// CHECK-NEXT:      vfmadd231ps ymm5, ymm11, ymm1
// CHECK-NEXT:      vfmadd231ps ymm6, ymm11, ymm2
// CHECK-NEXT:      vfmadd231ps ymm7, ymm11, ymm3
// CHECK-NEXT:      vmovups [rcx], ymm4
// CHECK-NEXT:      vmovups [rcx+32], ymm5
// CHECK-NEXT:      vmovups [rcx+64], ymm6
// CHECK-NEXT:      vmovups [rcx+96], ymm7
// CHECK-NEXT:      ret
