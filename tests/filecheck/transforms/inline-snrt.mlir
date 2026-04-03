// RUN: xdsl-opt %s -p inline-snrt{cluster-num=2} | filecheck %s


%global_core_base_hartid = "snrt.global_core_base_hartid"() : () -> i32
%global_core_idx = "snrt.global_core_idx"() : () -> i32
%global_core_num = "snrt.global_core_num"() : () -> i32
// unsupported: %global_compute_core_idx = "snrt.global_compute_core_idx"() : () -> i32
// unsupported: %global_compute_core_num = "snrt.global_compute_core_num"() : () -> i32
// unsupported: %global_dm_core_num = "snrt.global_dm_core_num"() : () -> i32
%gcluster_core_idx = "snrt.cluster_core_idx"() : () -> i32
%cluster_core_num = "snrt.cluster_core_num"() : () -> i32
// unsupported: %cluster_compute_core_idx = "snrt.cluster_compute_core_idx"() : () -> i32
%cluster_compute_core_num = "snrt.cluster_compute_core_num"() : () -> i32
// unsupported: %cluster_dm_core_idx = "snrt.cluster_dm_core_idx"() : () -> i32
%cluster_dm_core_num = "snrt.cluster_dm_core_num"() : () -> i32
%cluster_idx = "snrt.cluster_idx"() : () -> i32
%cluster_num = "snrt.cluster_num"() : () -> i32
%is_compute_core = "snrt.is_compute_core"() : () -> i1
%is_dm_core = "snrt.is_dm_core"() : () -> i1

"test.op"(%global_core_base_hartid, %global_core_idx, %global_core_num, %gcluster_core_idx, %cluster_core_num, %cluster_compute_core_num, %cluster_dm_core_num, %cluster_idx, %cluster_num, %is_compute_core, %is_dm_core) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1) -> ()

"snrt.cluster_hw_barrier"() : () -> ()
"snrt.ssr_disable"() : () -> ()

%dst, %src, %size = "test.op"() : () -> (i32, i32, i32)
%tx_id = "snrt.dma_start_1d"(%dst, %src, %size) : (i32, i32, i32) -> i32
"test.op"(%tx_id) : (i32) -> ()

%dst_wide, %src_wide = "test.op"() : () -> (i64, i64)
%tx_id2 = "snrt.dma_start_1d_wideptr"(%dst_wide, %src_wide, %size) : (i64, i64, i32) -> i32
"test.op"(%tx_id2) : (i32) -> ()

%dst_stride, %src_stride, %repeat = "test.op"() : () -> (i32, i32, i32)
%tx_id3 = "snrt.dma_start_2d_wideptr"(%dst_wide, %src_wide, %dst_stride, %src_stride, %size, %repeat) : (i64, i64, i32, i32, i32, i32) -> i32
"test.op"(%tx_id3) : (i32) -> ()

%tx_id4 = "snrt.dma_start_2d"(%dst, %src, %dst_stride, %src_stride, %size, %repeat) : (i32, i32, i32, i32, i32, i32) -> i32
"test.op"(%tx_id4) : (i32) -> ()


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %global_core_base_hartid = arith.constant 0 : i32
// CHECK-NEXT:   %global_core_idx = rv32.get_register : !riscv.reg<zero>
// CHECK-NEXT:   %global_core_idx_1 = riscv.csrrs %global_core_idx, -236, "r" : (!riscv.reg<zero>) -> !riscv.reg
// CHECK-NEXT:   %global_core_idx_2 = builtin.unrealized_conversion_cast %global_core_idx_1 : !riscv.reg to i32
// CHECK-NEXT:   %global_core_idx_3 = arith.constant 0 : i32
// CHECK-NEXT:   %global_core_idx_4 = arith.subi %global_core_idx_2, %global_core_idx_3 : i32
// CHECK-NEXT:   %global_core_num = arith.constant 18 : i32
// CHECK-NEXT:   %gcluster_core_idx = rv32.get_register : !riscv.reg<zero>
// CHECK-NEXT:   %gcluster_core_idx_1 = riscv.csrrs %gcluster_core_idx, -236, "r" : (!riscv.reg<zero>) -> !riscv.reg
// CHECK-NEXT:   %gcluster_core_idx_2 = builtin.unrealized_conversion_cast %gcluster_core_idx_1 : !riscv.reg to i32
// CHECK-NEXT:   %gcluster_core_idx_3 = arith.constant 0 : i32
// CHECK-NEXT:   %gcluster_core_idx_4 = arith.subi %gcluster_core_idx_2, %gcluster_core_idx_3 : i32
// CHECK-NEXT:   %gcluster_core_idx_5 = arith.constant 9 : i32
// CHECK-NEXT:   %gcluster_core_idx_6 = arith.remsi %gcluster_core_idx_4, %gcluster_core_idx_5 : i32
// CHECK-NEXT:   %cluster_core_num = arith.constant 9 : i32
// CHECK-NEXT:   %cluster_compute_core_num = arith.constant 8 : i32
// CHECK-NEXT:   %cluster_dm_core_num = arith.constant 1 : i32
// CHECK-NEXT:   %cluster_idx = arith.constant 9 : i32
// CHECK-NEXT:   %cluster_idx_1 = rv32.get_register : !riscv.reg<zero>
// CHECK-NEXT:   %cluster_idx_2 = riscv.csrrs %cluster_idx_1, -236, "r" : (!riscv.reg<zero>) -> !riscv.reg
// CHECK-NEXT:   %cluster_idx_3 = builtin.unrealized_conversion_cast %cluster_idx_2 : !riscv.reg to i32
// CHECK-NEXT:   %cluster_idx_4 = arith.constant 0 : i32
// CHECK-NEXT:   %cluster_idx_5 = arith.subi %cluster_idx_3, %cluster_idx_4 : i32
// CHECK-NEXT:   %cluster_idx_6 = arith.divsi %cluster_idx_5, %cluster_idx : i32
// CHECK-NEXT:   %cluster_num = arith.constant 2 : i32
// CHECK-NEXT:   %is_compute_core = rv32.get_register : !riscv.reg<zero>
// CHECK-NEXT:   %is_compute_core_1 = riscv.csrrs %is_compute_core, -236, "r" : (!riscv.reg<zero>) -> !riscv.reg
// CHECK-NEXT:   %is_compute_core_2 = builtin.unrealized_conversion_cast %is_compute_core_1 : !riscv.reg to i32
// CHECK-NEXT:   %is_compute_core_3 = arith.constant 0 : i32
// CHECK-NEXT:   %is_compute_core_4 = arith.subi %is_compute_core_2, %is_compute_core_3 : i32
// CHECK-NEXT:   %is_compute_core_5 = arith.constant 9 : i32
// CHECK-NEXT:   %is_compute_core_6 = arith.remsi %is_compute_core_4, %is_compute_core_5 : i32
// CHECK-NEXT:   %is_compute_core_7 = arith.constant 8 : i32
// CHECK-NEXT:   %is_compute_core_8 = arith.cmpi slt, %is_compute_core_6, %is_compute_core_7 : i32
// CHECK-NEXT:   %is_dm_core = rv32.get_register : !riscv.reg<zero>
// CHECK-NEXT:   %is_dm_core_1 = riscv.csrrs %is_dm_core, -236, "r" : (!riscv.reg<zero>) -> !riscv.reg
// CHECK-NEXT:   %is_dm_core_2 = builtin.unrealized_conversion_cast %is_dm_core_1 : !riscv.reg to i32
// CHECK-NEXT:   %is_dm_core_3 = arith.constant 0 : i32
// CHECK-NEXT:   %is_dm_core_4 = arith.subi %is_dm_core_2, %is_dm_core_3 : i32
// CHECK-NEXT:   %is_dm_core_5 = arith.constant 9 : i32
// CHECK-NEXT:   %is_dm_core_6 = arith.remsi %is_dm_core_4, %is_dm_core_5 : i32
// CHECK-NEXT:   %is_dm_core_7 = arith.constant 8 : i32
// CHECK-NEXT:   %is_dm_core_8 = arith.cmpi sge, %is_dm_core_6, %is_dm_core_7 : i32

// CHECK-NEXT:   "test.op"(%global_core_base_hartid, %global_core_idx_4, %global_core_num, %gcluster_core_idx_6, %cluster_core_num, %cluster_compute_core_num, %cluster_dm_core_num, %cluster_idx_6, %cluster_num, %is_compute_core_8, %is_dm_core_8) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1) -> ()


                 // Lowering of cluster_hw_barrier
// CHECK-NEXT:   %0 = rv32.get_register : !riscv.reg<zero>
// CHECK-NEXT:   %1 = riscv.csrrs %0, 1986 : (!riscv.reg<zero>) -> !riscv.reg<zero>

                 // Lowering of ssr_disable
// CHECK-NEXT:   %2 = riscv.csrrci 1984, 1 : () -> !riscv.reg
// CHECK-NEXT:   %dst, %src, %size = "test.op"() : () -> (i32, i32, i32)

                 // Lowering for dma_start_1d
// CHECK-NEXT:   %tx_id = rv32.get_register : !riscv.reg<zero>
// CHECK-NEXT:   %tx_id_1 = builtin.unrealized_conversion_cast %dst : i32 to !riscv.reg
// CHECK-NEXT:   %tx_id_2 = builtin.unrealized_conversion_cast %src : i32 to !riscv.reg
// CHECK-NEXT:   %tx_id_3 = builtin.unrealized_conversion_cast %size : i32 to !riscv.reg
// CHECK-NEXT:   riscv_snitch.dmsrc %tx_id_2, %tx_id : (!riscv.reg, !riscv.reg<zero>) -> ()
// CHECK-NEXT:   riscv_snitch.dmdst %tx_id_1, %tx_id : (!riscv.reg, !riscv.reg<zero>) -> ()
// CHECK-NEXT:   %tx_id_4 = riscv_snitch.dmcpyi %tx_id_3, 0 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %tx_id_5 = builtin.unrealized_conversion_cast %tx_id_4 : !riscv.reg to i32

// CHECK-NEXT:   "test.op"(%tx_id_5) : (i32) -> ()

// CHECK-NEXT:   %dst_wide, %src_wide = "test.op"() : () -> (i64, i64)

                 // Lowering of dma_start_1d_wideptr
// CHECK-NEXT:   %tx_id2, %tx_id2_1 = builtin.unrealized_conversion_cast %dst_wide : i64 to !riscv.reg, !riscv.reg
// CHECK-NEXT:   %tx_id2_2, %tx_id2_3 = builtin.unrealized_conversion_cast %src_wide : i64 to !riscv.reg, !riscv.reg
// CHECK-NEXT:   %tx_id2_4 = builtin.unrealized_conversion_cast %size : i32 to !riscv.reg
// CHECK-NEXT:   riscv_snitch.dmsrc %tx_id2_2, %tx_id2_3 : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:   riscv_snitch.dmdst %tx_id2, %tx_id2_1 : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:   %tx_id2_5 = riscv_snitch.dmcpyi %tx_id2_4, 0 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %tx_id2_6 = builtin.unrealized_conversion_cast %tx_id2_5 : !riscv.reg to i32

// CHECK-NEXT:   "test.op"(%tx_id2_6) : (i32) -> ()

// CHECK-NEXT:   %dst_stride, %src_stride, %repeat = "test.op"() : () -> (i32, i32, i32)

                 // Lowering for dma_start_2d_wideptr
// CHECK-NEXT:   %3, %4 = builtin.unrealized_conversion_cast %dst_wide : i64 to !riscv.reg, !riscv.reg
// CHECK-NEXT:   %5, %6 = builtin.unrealized_conversion_cast %src_wide : i64 to !riscv.reg, !riscv.reg
// CHECK-NEXT:   %7 = builtin.unrealized_conversion_cast %src_stride : i32 to !riscv.reg
// CHECK-NEXT:   %8 = builtin.unrealized_conversion_cast %dst_stride : i32 to !riscv.reg
// CHECK-NEXT:   %9 = builtin.unrealized_conversion_cast %size : i32 to !riscv.reg
// CHECK-NEXT:   %10 = builtin.unrealized_conversion_cast %size : i32 to !riscv.reg
// CHECK-NEXT:   riscv_snitch.dmsrc %5, %6 : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:   riscv_snitch.dmdst %3, %4 : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:   riscv_snitch.dmstr %7, %8 : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:   riscv_snitch.dmrep %10 : (!riscv.reg) -> ()
// CHECK-NEXT:   %tx_id3 = riscv_snitch.dmcpyi %9, 2 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %tx_id3_1 = builtin.unrealized_conversion_cast %tx_id3 : !riscv.reg to i32

// CHECK-NEXT:   "test.op"(%tx_id3_1) : (i32) -> ()

                 // Lowering for dma_start_2d
// CHECK-NEXT:   %11 = rv32.get_register : !riscv.reg<zero>
// CHECK-NEXT:   %12 = builtin.unrealized_conversion_cast %dst : i32 to !riscv.reg
// CHECK-NEXT:   %13 = builtin.unrealized_conversion_cast %src : i32 to !riscv.reg
// CHECK-NEXT:   %14 = builtin.unrealized_conversion_cast %src_stride : i32 to !riscv.reg
// CHECK-NEXT:   %15 = builtin.unrealized_conversion_cast %dst_stride : i32 to !riscv.reg
// CHECK-NEXT:   %16 = builtin.unrealized_conversion_cast %size : i32 to !riscv.reg
// CHECK-NEXT:   %17 = builtin.unrealized_conversion_cast %size : i32 to !riscv.reg
// CHECK-NEXT:   riscv_snitch.dmsrc %13, %11 : (!riscv.reg, !riscv.reg<zero>) -> ()
// CHECK-NEXT:   riscv_snitch.dmdst %12, %11 : (!riscv.reg, !riscv.reg<zero>) -> ()
// CHECK-NEXT:   riscv_snitch.dmstr %14, %15 : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:   riscv_snitch.dmrep %17 : (!riscv.reg) -> ()
// CHECK-NEXT:   %tx_id4 = riscv_snitch.dmcpyi %16, 2 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %tx_id4_1 = builtin.unrealized_conversion_cast %tx_id4 : !riscv.reg to i32
// CHECK-NEXT:   "test.op"(%tx_id4_1) : (i32) -> ()
// CHECK-NEXT: }
