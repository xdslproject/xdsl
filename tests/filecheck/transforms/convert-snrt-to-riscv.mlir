// RUN: xdsl-opt %s -p convert-snrt-to-riscv | filecheck %s

"snrt.cluster_hw_barrier"() : () -> ()
"snrt.ssr_disable"() : () -> ()

%dst, %src, %size = "test.op"() : () -> (i32, i32, i32)
%tx_id = "snrt.dma_start_1d"(%dst, %src, %size) : (i32, i32, i32) -> i32

%dst_wide, %src_wide = "test.op"() : () -> (i64, i64)
%tx_id2 = "snrt.dma_start_1d_wideptr"(%dst_wide, %src_wide, %size) : (i64, i64, i32) -> i32


%dst_stride, %src_stride, %repeat = "test.op"() : () -> (i32, i32, i32)
%tx_id3 = "snrt.dma_start_2d_wideptr"(%dst_wide, %src_wide, %dst_stride, %src_stride, %size, %repeat) : (i64, i64, i32, i32, i32, i32) -> i32

%tx_id4 = "snrt.dma_start_2d"(%dst, %src, %dst_stride, %src_stride, %size, %repeat) : (i32, i32, i32, i32, i32, i32) -> i32


// CHECK-NEXT: builtin.module {

                 // Lowering of cluster_hw_barrier
// CHECK-NEXT:   %0 = riscv.get_register : () -> !riscv.reg<zero>
// CHECK-NEXT:   %1 = riscv.csrrs %0, 1986 : (!riscv.reg<zero>) -> !riscv.reg<zero>

                 // Lowering of ssr_disable
// CHECK-NEXT:   %2 = riscv.csrrci 1984, 1 : () -> !riscv.reg<>
// CHECK-NEXT:   %dst, %src, %size = "test.op"() : () -> (i32, i32, i32)

                 // Lowering for dma_start_1d
// CHECK-NEXT:   %tx_id = riscv.get_register : () -> !riscv.reg<zero>
// CHECK-NEXT:   %tx_id_1 = builtin.unrealized_conversion_cast %dst : i32 to !riscv.reg<>
// CHECK-NEXT:   %tx_id_2 = builtin.unrealized_conversion_cast %src : i32 to !riscv.reg<>
// CHECK-NEXT:   %tx_id_3 = builtin.unrealized_conversion_cast %size : i32 to !riscv.reg<>
// CHECK-NEXT:   riscv_snitch.dmsrc %tx_id_2, %tx_id : (!riscv.reg<>, !riscv.reg<zero>) -> ()
// CHECK-NEXT:   riscv_snitch.dmdst %tx_id_1, %tx_id : (!riscv.reg<>, !riscv.reg<zero>) -> ()
// CHECK-NEXT:   %tx_id_4 = riscv_snitch.dmcpyi %tx_id_3, 0 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %tx_id_5 = builtin.unrealized_conversion_cast %tx_id_4 : !riscv.reg<> to i32

// CHECK-NEXT:   %dst_wide, %src_wide = "test.op"() : () -> (i64, i64)

                 // Lowering of dma_start_1d_wideptr
// CHECK-NEXT:   %tx_id2, %tx_id2_1 = builtin.unrealized_conversion_cast %dst_wide : i64 to !riscv.reg<>, !riscv.reg<>
// CHECK-NEXT:   %tx_id2_2, %tx_id2_3 = builtin.unrealized_conversion_cast %src_wide : i64 to !riscv.reg<>, !riscv.reg<>
// CHECK-NEXT:   %tx_id2_4 = builtin.unrealized_conversion_cast %size : i32 to !riscv.reg<>
// CHECK-NEXT:   riscv_snitch.dmsrc %tx_id2_2, %tx_id2_3 : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:   riscv_snitch.dmdst %tx_id2, %tx_id2_1 : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:   %tx_id2_5 = riscv_snitch.dmcpyi %tx_id2_4, 0 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %tx_id2_6 = builtin.unrealized_conversion_cast %tx_id2_5 : !riscv.reg<> to i32

// CHECK-NEXT:   %dst_stride, %src_stride, %repeat = "test.op"() : () -> (i32, i32, i32)

                 // Lowering for dma_start_2d_wideptr
// CHECK-NEXT:   %3, %4 = builtin.unrealized_conversion_cast %dst_wide : i64 to !riscv.reg<>, !riscv.reg<>
// CHECK-NEXT:   %5, %6 = builtin.unrealized_conversion_cast %src_wide : i64 to !riscv.reg<>, !riscv.reg<>
// CHECK-NEXT:   %7 = builtin.unrealized_conversion_cast %src_stride : i32 to !riscv.reg<>
// CHECK-NEXT:   %8 = builtin.unrealized_conversion_cast %dst_stride : i32 to !riscv.reg<>
// CHECK-NEXT:   %9 = builtin.unrealized_conversion_cast %size : i32 to !riscv.reg<>
// CHECK-NEXT:   %10 = builtin.unrealized_conversion_cast %size : i32 to !riscv.reg<>
// CHECK-NEXT:   riscv_snitch.dmsrc %5, %6 : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:   riscv_snitch.dmdst %3, %4 : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:   riscv_snitch.dmstr %7, %8 : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:   riscv_snitch.dmrep %10 : (!riscv.reg<>) -> ()
// CHECK-NEXT:   %tx_id3 = riscv_snitch.dmcpyi %9, 2 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %tx_id3_1 = builtin.unrealized_conversion_cast %tx_id3 : !riscv.reg<> to i32

                 // Lowering for dma_start_2d
// CHECK-NEXT:   %11 = riscv.get_register : () -> !riscv.reg<zero>
// CHECK-NEXT:   %12 = builtin.unrealized_conversion_cast %dst : i32 to !riscv.reg<>
// CHECK-NEXT:   %13 = builtin.unrealized_conversion_cast %src : i32 to !riscv.reg<>
// CHECK-NEXT:   %14 = builtin.unrealized_conversion_cast %src_stride : i32 to !riscv.reg<>
// CHECK-NEXT:   %15 = builtin.unrealized_conversion_cast %dst_stride : i32 to !riscv.reg<>
// CHECK-NEXT:   %16 = builtin.unrealized_conversion_cast %size : i32 to !riscv.reg<>
// CHECK-NEXT:   %17 = builtin.unrealized_conversion_cast %size : i32 to !riscv.reg<>
// CHECK-NEXT:   riscv_snitch.dmsrc %13, %11 : (!riscv.reg<>, !riscv.reg<zero>) -> ()
// CHECK-NEXT:   riscv_snitch.dmdst %12, %11 : (!riscv.reg<>, !riscv.reg<zero>) -> ()
// CHECK-NEXT:   riscv_snitch.dmstr %14, %15 : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:   riscv_snitch.dmrep %17 : (!riscv.reg<>) -> ()
// CHECK-NEXT:   %tx_id4 = riscv_snitch.dmcpyi %16, 2 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %tx_id4_1 = builtin.unrealized_conversion_cast %tx_id4 : !riscv.reg<> to i32
// CHECK-NEXT: }
