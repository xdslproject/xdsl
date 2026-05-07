// RUN: xdsl-opt -p riscv-legalize-parallel-mov --split-input-file --verify-diagnostics %s | filecheck %s

builtin.module {
    riscv_func.func @main() {
        %0, %1, %2 = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)
        %3 = riscv.parallel_mov %0 [32] : (!riscv.reg) -> (!riscv.reg)
        %4 = riscv.add %2, %3 : (!riscv.reg, !riscv.reg) -> (!riscv.reg)
        riscv_func.return
    }
}
// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() {
// CHECK-NEXT:      %0, %1, %2 = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)
// CHECK-NEXT:      %3, %4 = riscv.parallel_mov %0, %2 [32, 32] : (!riscv.reg, !riscv.reg) -> (!riscv.reg, !riscv.reg)
// CHECK-NEXT:      %5 = riscv.add %4, %3 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
