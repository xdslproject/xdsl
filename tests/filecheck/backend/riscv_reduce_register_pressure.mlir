// RUN: xdsl-opt -p riscv-reduce-register-pressure %s | filecheck %s

builtin.module {
  // Replace loading immediate value 0
  %0 = riscv.li 0 : () -> !riscv.reg<>
  %1 = riscv.li 0 : () -> !riscv.reg<a0>
  %2 = riscv.li 1 : () -> !riscv.reg<>
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %{{.*}} = riscv.get_register : () -> !riscv.reg<zero>
// CHECK-NEXT:   %{{.*}} = riscv.get_register : () -> !riscv.reg<zero>
// CHECK-NEXT:   %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<zero>) -> !riscv.reg<a0>
// CHECK-NEXT:   %{{.*}} = riscv.li 1 : () -> !riscv.reg<>
// CHECK-NEXT: }
