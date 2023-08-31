// RUN: xdsl-opt -p riscv-reduce-register-pressure %s | filecheck %s

builtin.module {
  %0 = riscv.li 0 : () -> !riscv.reg<>
  %1 = riscv.li 0 : () -> !riscv.reg<a0>
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = riscv.get_register : () -> !riscv.reg<zero>
// CHECK-NEXT:   %1 = riscv.get_register : () -> !riscv.reg<zero>
// CHECK-NEXT:   %2 = riscv.mv %1 : (!riscv.reg<zero>) -> !riscv.reg<a0>
// CHECK-NEXT: }
