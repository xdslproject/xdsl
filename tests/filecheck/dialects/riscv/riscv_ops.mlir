// RUN: xdsl-opt %s | xdsl-opt | filecheck %s
"builtin.module"() ({
  %0 = "test.op"() : () -> !riscv.reg<>
  %1 = "test.op"() : () -> !riscv.reg<>
  %2 = "riscv.add"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK: %2 = "riscv.add"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %3 = "riscv.sub"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %3 = "riscv.sub"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %4 = "riscv.li"() {"immediate" = 1 : i32}: () -> !riscv.reg<>
  // CHECK-NEXT: %4 = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<>
}) : () -> ()
