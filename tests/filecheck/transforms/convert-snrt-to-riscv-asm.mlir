// RUN: xdsl-opt %s -p convert-snrt-to-riscv-asm | filecheck %s

"snrt.cluster_hw_barrier"() : () -> ()

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %0 = riscv.get_register : () -> !riscv.reg<zero>
// CHECK-NEXT:   %1 = riscv.csrrs %0, 1986 : (!riscv.reg<zero>) -> !riscv.reg<zero>
// CHECK-NEXT: }
