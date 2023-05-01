// RUN: xdsl-opt %s | xdsl-opt | filecheck %s
"builtin.module"() ({
  %lhs = "test.op"() : () -> !riscv.reg<>
  %rhs = "test.op"() : () -> !riscv.reg<>
  %sum = "riscv.add"(%lhs, %rhs) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
}) : () -> ()

// CHECK:       "builtin.module"() ({
// CHECK-NEXT:       %lhs = "test.op"() : () -> !riscv.reg<>
// CHECK-NEXT:       %rhs = "test.op"() : () -> !riscv.reg<>
// CHECK-NEXT:       %sum = "riscv.add"(%lhs, %rhs) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   }) : () -> ()
