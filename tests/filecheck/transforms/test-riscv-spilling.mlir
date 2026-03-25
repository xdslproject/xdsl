// RUN: xdsl-opt --split-input-file -p test-riscv-spilling %s | filecheck %s
builtin.module {
  riscv_func.func @main() {
    %0, %1, %2 = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)
    %3 = riscv.add %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %4 = riscv.add %2, %3 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %5 = riscv.add %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    riscv_func.return
  }
}
// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() {
// CHECK-NEXT:      %0 = rv32.get_register : !riscv.reg<sp>
// CHECK-NEXT:      %1 = riscv.addi %0, -16 : (!riscv.reg<sp>) -> !riscv.reg<sp>
// CHECK-NEXT:      %2, %3, %4 = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)
// CHECK-NEXT:      riscv.sw %0, %4, 0 : (!riscv.reg<sp>, !riscv.reg) -> ()
// CHECK-NEXT:      %5 = riscv.add %2, %3 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:      riscv.sw %0, %2, 4 : (!riscv.reg<sp>, !riscv.reg) -> ()
// CHECK-NEXT:      %6 = riscv.lw %0, 0 : (!riscv.reg<sp>) -> !riscv.reg
// CHECK-NEXT:      %7 = riscv.add %6, %5 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:      riscv.sw %0, %6, 8 : (!riscv.reg<sp>, !riscv.reg) -> ()
// CHECK-NEXT:      %8 = riscv.lw %0, 4 : (!riscv.reg<sp>) -> !riscv.reg
// CHECK-NEXT:      %9 = riscv.add %8, %3 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:      %10 = riscv.addi %0, 16 : (!riscv.reg<sp>) -> !riscv.reg<sp>
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
