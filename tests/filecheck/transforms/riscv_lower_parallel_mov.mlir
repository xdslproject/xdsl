// RUN: xdsl-opt -p riscv-lower-parallel-mov %s | filecheck %s

builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
  %2, %3 = riscv.parallel_mov %0, %1 : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg<s1>, !riscv.reg<s2>)
  %4 = riscv.add %2, %3 : (!riscv.reg<s1>, !riscv.reg<s2>) -> !riscv.reg
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
// CHECK-NEXT:    %2 = riscv.add %0, %1 : (!riscv.reg<s1>, !riscv.reg<s2>) -> !riscv.reg
// CHECK-NEXT:  }
