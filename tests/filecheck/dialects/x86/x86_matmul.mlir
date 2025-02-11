// RUN: XDSL_ROUNDTRIP

// C: 4x2xf32 = A: 4x8xf32 * B: 8x2xf32
x86_func.func @matmul() {
    // Scalar registers loading
    %rdi = x86.get_register : () -> !x86.reg<rdi>
    %rsi = x86.get_register : () -> !x86.reg<rsi>
    %rcx = x86.get_register : () -> !x86.reg<rcx>
    // Vector registers loading
    %ymm0 = x86.get_avx_register : () -> !x86.avx2reg<ymm0>
    %ymm1 = x86.get_avx_register : () -> !x86.avx2reg<ymm1>
    %ymm2 = x86.get_avx_register : () -> !x86.avx2reg<ymm2>
    %ymm3 = x86.get_avx_register : () -> !x86.avx2reg<ymm3>
    %ymm4 = x86.get_avx_register : () -> !x86.avx2reg<ymm4>
    %ymm5 = x86.get_avx_register : () -> !x86.avx2reg<ymm5>
    %ymm6 = x86.get_avx_register : () -> !x86.avx2reg<ymm6>
    %ymm7 = x86.get_avx_register : () -> !x86.avx2reg<ymm7>
    %ymm8 = x86.get_avx_register : () -> !x86.avx2reg<ymm8>
    // Load rows of A
    %a_row_0 = x86.rm.vmovups %ymm0, %rdi, 0 : (!x86.avx2reg<ymm0>, !x86.reg<rdi>) -> (!x86.avx2reg<ymm0>)
    %a_row_1 = x86.rm.vmovups %ymm1, %rdi, 32 : (!x86.avx2reg<ymm1>, !x86.reg<rdi>) -> (!x86.avx2reg<ymm1>)
    %a_row_2 = x86.rm.vmovups %ymm2, %rdi, 64 : (!x86.avx2reg<ymm2>, !x86.reg<rdi>) -> (!x86.avx2reg<ymm2>)
    %a_row_3 = x86.rm.vmovups %ymm3, %rdi, 96 : (!x86.avx2reg<ymm3>, !x86.reg<rdi>) -> (!x86.avx2reg<ymm3>)
    // Initialize the accumulators (rows of C)
    %c0_tmp0 = x86.rm.vmovups %ymm4, %rcx, 0 : (!x86.avx2reg<ymm4>, !x86.reg<rcx>) -> !x86.avx2reg<ymm4>
    %c1_tmp0 = x86.rm.vmovups %ymm5, %rcx, 32 : (!x86.avx2reg<ymm5>, !x86.reg<rcx>) -> !x86.avx2reg<ymm5>
    %c2_tmp0 = x86.rm.vmovups %ymm6, %rcx, 64 : (!x86.avx2reg<ymm6>, !x86.reg<rcx>) -> !x86.avx2reg<ymm6>
    %c3_tmp0 = x86.rm.vmovups %ymm7, %rcx, 96 : (!x86.avx2reg<ymm7>, !x86.reg<rcx>) -> !x86.avx2reg<ymm7>
    // Load column 0 of B
    %b_col_0 = x86.rm.vbroadcastss %ymm8, %rsi, 0 : (!x86.avx2reg<ymm8>, !x86.reg<rsi>) -> !x86.avx2reg<ymm8>
    // Go brrr
    %c0_tmp1 = x86.rrr.vfmadd231pd %b_col_0, %a_row_0, %c0_tmp0 : (!x86.avx2reg<ymm8>, !x86.avx2reg<ymm0>, !x86.avx2reg<ymm4>) -> !x86.avx2reg<ymm4>
    %c1_tmp1 = x86.rrr.vfmadd231pd %b_col_0, %a_row_1, %c1_tmp0 : (!x86.avx2reg<ymm8>, !x86.avx2reg<ymm1>, !x86.avx2reg<ymm5>) -> !x86.avx2reg<ymm5>
    %c2_tmp1 = x86.rrr.vfmadd231pd %b_col_0, %a_row_2, %c2_tmp0 : (!x86.avx2reg<ymm8>, !x86.avx2reg<ymm2>, !x86.avx2reg<ymm6>) -> !x86.avx2reg<ymm6>
    %c3_tmp1 = x86.rrr.vfmadd231pd %b_col_0, %a_row_3, %c3_tmp0 : (!x86.avx2reg<ymm8>, !x86.avx2reg<ymm3>, !x86.avx2reg<ymm7>) -> !x86.avx2reg<ymm7>
    // Load column 1 of B
    %b_col_1 = x86.rm.vbroadcastss %ymm8, %rsi, 4 : (!x86.avx2reg<ymm8>, !x86.reg<rsi>) -> !x86.avx2reg<ymm8>
    // Go brrr
    %c0 = x86.rrr.vfmadd231pd %b_col_1, %a_row_0, %c0_tmp1 : (!x86.avx2reg<ymm8>, !x86.avx2reg<ymm0>, !x86.avx2reg<ymm4>) -> !x86.avx2reg<ymm4>
    %c1 = x86.rrr.vfmadd231pd %b_col_1, %a_row_1, %c1_tmp1 : (!x86.avx2reg<ymm8>, !x86.avx2reg<ymm1>, !x86.avx2reg<ymm5>) -> !x86.avx2reg<ymm5>
    %c2 = x86.rrr.vfmadd231pd %b_col_1, %a_row_2, %c2_tmp1 : (!x86.avx2reg<ymm8>, !x86.avx2reg<ymm2>, !x86.avx2reg<ymm6>) -> !x86.avx2reg<ymm6>
    %c3 = x86.rrr.vfmadd231pd %b_col_1, %a_row_3, %c3_tmp1 : (!x86.avx2reg<ymm8>, !x86.avx2reg<ymm3>, !x86.avx2reg<ymm7>) -> !x86.avx2reg<ymm7>
   // Store the results (rows of C)
    x86.mr.vmovups %rcx, %c0, 0 : (!x86.reg<rcx>,!x86.avx2reg<ymm4>) -> ()
    x86.mr.vmovups %rcx, %c1, 32 : (!x86.reg<rcx>,!x86.avx2reg<ymm5>) -> ()
    x86.mr.vmovups %rcx,%c2,64 : (!x86.reg<rcx>,!x86.avx2reg<ymm6>) -> ()
    x86.mr.vmovups %rcx,%c3,96 : (!x86.reg<rcx>,!x86.avx2reg<ymm7>) -> ()
    // Boom
    x86_func.ret
}

// CHECK:      builtin.module {
// CHECK-NEXT:     x86_func.func @matmul() {
// CHECK-NEXT:     %{{.*}} = x86.get_register : () -> !x86.reg<rdi>
// CHECK-NEXT:     %{{.*}} = x86.get_register : () -> !x86.reg<rsi>
// CHECK-NEXT:     %{{.*}} = x86.get_register : () -> !x86.reg<rcx>
// CHECK-NEXT:     %{{.*}} = x86.get_avx_register : () -> !x86.avx2reg<ymm0>
// CHECK-NEXT:     %{{.*}} = x86.get_avx_register : () -> !x86.avx2reg<ymm1>
// CHECK-NEXT:     %{{.*}} = x86.get_avx_register : () -> !x86.avx2reg<ymm2>
// CHECK-NEXT:     %{{.*}} = x86.get_avx_register : () -> !x86.avx2reg<ymm3>
// CHECK-NEXT:     %{{.*}} = x86.get_avx_register : () -> !x86.avx2reg<ymm4>
// CHECK-NEXT:     %{{.*}} = x86.get_avx_register : () -> !x86.avx2reg<ymm5>
// CHECK-NEXT:     %{{.*}} = x86.get_avx_register : () -> !x86.avx2reg<ymm6>
// CHECK-NEXT:     %{{.*}} = x86.get_avx_register : () -> !x86.avx2reg<ymm7>
// CHECK-NEXT:     %{{.*}} = x86.get_avx_register : () -> !x86.avx2reg<ymm8>
// CHECK-NEXT:     %{{.*}}_row = x86.rm.vmovups %{{.*}}, %{{.*}}, 0 : (!x86.avx2reg<ymm0>, !x86.reg<rdi>) -> !x86.avx2reg<ymm0>
// CHECK-NEXT:     %{{.*}}_row_1 = x86.rm.vmovups %{{.*}}, %{{.*}}, 32 : (!x86.avx2reg<ymm1>, !x86.reg<rdi>) -> !x86.avx2reg<ymm1>
// CHECK-NEXT:     %{{.*}}_row_2 = x86.rm.vmovups %{{.*}}, %{{.*}}, 64 : (!x86.avx2reg<ymm2>, !x86.reg<rdi>) -> !x86.avx2reg<ymm2>
// CHECK-NEXT:     %{{.*}}_row_3 = x86.rm.vmovups %{{.*}}, %{{.*}}, 96 : (!x86.avx2reg<ymm3>, !x86.reg<rdi>) -> !x86.avx2reg<ymm3>
// CHECK-NEXT:     %{{.*}}_tmp0 = x86.rm.vmovups %{{.*}}, %{{.*}}, 0 : (!x86.avx2reg<ymm4>, !x86.reg<rcx>) -> !x86.avx2reg<ymm4>
// CHECK-NEXT:     %{{.*}}_tmp0 = x86.rm.vmovups %{{.*}}, %{{.*}}, 32 : (!x86.avx2reg<ymm5>, !x86.reg<rcx>) -> !x86.avx2reg<ymm5>
// CHECK-NEXT:     %{{.*}}_tmp0 = x86.rm.vmovups %{{.*}}, %{{.*}}, 64 : (!x86.avx2reg<ymm6>, !x86.reg<rcx>) -> !x86.avx2reg<ymm6>
// CHECK-NEXT:     %{{.*}}_tmp0 = x86.rm.vmovups %{{.*}}, %{{.*}}, 96 : (!x86.avx2reg<ymm7>, !x86.reg<rcx>) -> !x86.avx2reg<ymm7>
// CHECK-NEXT:     %{{.*}}_col = x86.rm.vbroadcastss %{{.*}}, %{{.*}}, 0 : (!x86.avx2reg<ymm8>, !x86.reg<rsi>) -> !x86.avx2reg<ymm8>
// CHECK-NEXT:     %{{.*}}_tmp1 = x86.rrr.vfmadd231pd %{{.*}}_col, %{{.*}}_row, %{{.*}}_tmp0 : (!x86.avx2reg<ymm8>, !x86.avx2reg<ymm0>, !x86.avx2reg<ymm4>) -> !x86.avx2reg<ymm4>
// CHECK-NEXT:     %{{.*}}_tmp1 = x86.rrr.vfmadd231pd %{{.*}}_col, %{{.*}}_row_1, %{{.*}}_tmp0 : (!x86.avx2reg<ymm8>, !x86.avx2reg<ymm1>, !x86.avx2reg<ymm5>) -> !x86.avx2reg<ymm5>
// CHECK-NEXT:     %{{.*}}_tmp1 = x86.rrr.vfmadd231pd %{{.*}}_col, %{{.*}}_row_2, %{{.*}}_tmp0 : (!x86.avx2reg<ymm8>, !x86.avx2reg<ymm2>, !x86.avx2reg<ymm6>) -> !x86.avx2reg<ymm6>
// CHECK-NEXT:     %{{.*}}_tmp1 = x86.rrr.vfmadd231pd %{{.*}}_col, %{{.*}}_row_3, %{{.*}}_tmp0 : (!x86.avx2reg<ymm8>, !x86.avx2reg<ymm3>, !x86.avx2reg<ymm7>) -> !x86.avx2reg<ymm7>
// CHECK-NEXT:     %{{.*}}_col_1 = x86.rm.vbroadcastss %{{.*}}, %{{.*}}, 4 : (!x86.avx2reg<ymm8>, !x86.reg<rsi>) -> !x86.avx2reg<ymm8>
// CHECK-NEXT:     %{{.*}} = x86.rrr.vfmadd231pd %{{.*}}_col_1, %{{.*}}_row, %{{.*}}_tmp1 : (!x86.avx2reg<ymm8>, !x86.avx2reg<ymm0>, !x86.avx2reg<ymm4>) -> !x86.avx2reg<ymm4>
// CHECK-NEXT:     %{{.*}} = x86.rrr.vfmadd231pd %{{.*}}_col_1, %{{.*}}_row_1, %{{.*}}_tmp1 : (!x86.avx2reg<ymm8>, !x86.avx2reg<ymm1>, !x86.avx2reg<ymm5>) -> !x86.avx2reg<ymm5>
// CHECK-NEXT:     %{{.*}} = x86.rrr.vfmadd231pd %{{.*}}_col_1, %{{.*}}_row_2, %{{.*}}_tmp1 : (!x86.avx2reg<ymm8>, !x86.avx2reg<ymm2>, !x86.avx2reg<ymm6>) -> !x86.avx2reg<ymm6>
// CHECK-NEXT:     %{{.*}} = x86.rrr.vfmadd231pd %{{.*}}_col_1, %{{.*}}_row_3, %{{.*}}_tmp1 : (!x86.avx2reg<ymm8>, !x86.avx2reg<ymm3>, !x86.avx2reg<ymm7>) -> !x86.avx2reg<ymm7>
// CHECK-NEXT:     x86.mr.vmovups %{{.*}}, %{{.*}}, 0 : (!x86.reg<rcx>, !x86.avx2reg<ymm4>) -> ()
// CHECK-NEXT:     x86.mr.vmovups %{{.*}}, %{{.*}}, 32 : (!x86.reg<rcx>, !x86.avx2reg<ymm5>) -> ()
// CHECK-NEXT:     x86.mr.vmovups %{{.*}}, %{{.*}}, 64 : (!x86.reg<rcx>, !x86.avx2reg<ymm6>) -> ()
// CHECK-NEXT:     x86.mr.vmovups %{{.*}}, %{{.*}}, 96 : (!x86.reg<rcx>, !x86.avx2reg<ymm7>) -> ()
// CHECK-NEXT:     x86_func.ret
// CHECK-NEXT:   }
// }
