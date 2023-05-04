// RUN: xdsl-opt %s | xdsl-opt | filecheck %s
"builtin.module"() ({
  %0 = "test.op"() : () -> !riscv.reg<a1>
  %1 = "test.op"() : () -> !riscv.reg<>
  // add a0, a1, x?
  %add = "riscv.add"(%0, %1) : (!riscv.reg<a1>, !riscv.reg<>) -> !riscv.reg<a0>
// CHECK: %{{.*}} = "riscv.add"(%{{.*}}, %{{.*}}) : (!riscv.reg<a1>, !riscv.reg<>) -> !riscv.reg<a0>
}) : () -> ()
