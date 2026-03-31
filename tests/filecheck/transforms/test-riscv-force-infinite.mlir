// RUN: xdsl-opt --split-input-file -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive force_infinite=true}" %s | filecheck %s

// check that infinite registers are used
riscv_func.func @main() {
    %0, %1 = "test.op"() : () -> (!riscv.reg, !riscv.reg)
    %2, %3 = riscv.parallel_mov %0, %1 [32, 32] : (!riscv.reg, !riscv.reg) -> (!riscv.reg, !riscv.reg)
    riscv_func.return
}
// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() {
// CHECK-NEXT:      %0, %1 = "test.op"() : () -> (!riscv.reg<j_0>, !riscv.reg<j_1>)
// CHECK-NEXT:      %2, %3 = riscv.parallel_mov %0, %1 [32, 32] : (!riscv.reg<j_0>, !riscv.reg<j_1>) -> (!riscv.reg<j_0>, !riscv.reg<j_1>)
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
