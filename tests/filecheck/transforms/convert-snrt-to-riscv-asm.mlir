// RUN: xdsl-opt %s -p convert-snrt-to-riscv-asm | filecheck %s

"snrt.cluster_hw_barrier"() : () -> ()
"snrt.ssr_disable"() : () -> ()
%dst, %src, %size = "test.op"() : () -> (i32, i32, i32)
%tx_id = "snrt.dma_start_1d"(%dst, %src, %size) : (i32, i32, i32) -> i32

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %0 = riscv.get_register : () -> !riscv.reg<zero>
// CHECK-NEXT:   %1 = riscv.csrrs %0, 1986 : (!riscv.reg<zero>) -> !riscv.reg<zero>
// CHECK-NEXT:   %2 = riscv.csrrci 1984, 1 : () -> !riscv.reg<>
// CHECK-NEXT:   %dst, %src, %size = "test.op"() : () -> (i32, i32, i32)
// CHECK-NEXT:   %tx_id = riscv.get_register : () -> !riscv.reg<zero>
// CHECK-NEXT:   %tx_id_1 = builtin.unrealized_conversion_cast %dst : i32 to !riscv.reg<>
// CHECK-NEXT:   %tx_id_2 = builtin.unrealized_conversion_cast %src : i32 to !riscv.reg<>
// CHECK-NEXT:   %tx_id_3 = builtin.unrealized_conversion_cast %size : i32 to !riscv.reg<>
// CHECK-NEXT:   riscv_snitch.dmsrc %tx_id_2, %tx_id : (!riscv.reg<>, !riscv.reg<zero>) -> ()
// CHECK-NEXT:   riscv_snitch.dmdst %tx_id_1, %tx_id : (!riscv.reg<>, !riscv.reg<zero>) -> ()
// CHECK-NEXT:   %tx_id_4 = riscv_snitch.dmcpyi %tx_id_3, 0 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %tx_id_5 = builtin.unrealized_conversion_cast %tx_id_4 : !riscv.reg<> to i32
// CHECK-NEXT: }
