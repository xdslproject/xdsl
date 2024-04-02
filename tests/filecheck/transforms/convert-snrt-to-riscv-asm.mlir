// RUN: xdsl-opt %s -p convert-snrt-to-riscv-asm | filecheck %s

"snrt.cluster_hw_barrier"() : () -> ()
"snrt.ssr_disable"() : () -> ()

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %0 = riscv.get_register : () -> !riscv.reg<zero>
// CHECK-NEXT:   %1 = riscv.csrrs %0, 1986 : (!riscv.reg<zero>) -> !riscv.reg<zero>
// CHECK-NEXT:   %2 = riscv.csrrci 1984, 1 : () -> !riscv.reg<>
// CHECK-NEXT: }
