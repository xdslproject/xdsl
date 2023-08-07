// RUN: xdsl-opt -p optimise-riscv %s | filecheck %s

builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>)
  %2 = riscv.mv %0 : (!riscv.reg<a0>) -> !riscv.reg<a0>
  %3 = riscv.mv %1 : (!riscv.reg<a1>) -> !riscv.reg<a2>
}

// CHECK: builtin.module {
// CHECK-NEXT:   %0, %1 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>)
// CHECK-NEXT:   %2 = riscv.mv %1 : (!riscv.reg<a1>) -> !riscv.reg<a2>
// CHECK-NEXT: }
