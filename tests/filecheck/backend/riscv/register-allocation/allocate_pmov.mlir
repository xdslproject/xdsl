// RUN: xdsl-opt --split-input-file -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive}" %s | filecheck %s

// Check case where all registers are unallocated
riscv_func.func @main() {
    %0, %1, %2 = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)
    %3, %4, %5 = riscv.parallel_mov %0, %1, %2 [32, 32, 32] : (!riscv.reg, !riscv.reg, !riscv.reg) -> (!riscv.reg, !riscv.reg, !riscv.reg)
    "test.op"(%3, %4, %5) : (!riscv.reg, !riscv.reg, !riscv.reg) -> ()
    riscv_func.return
}

// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() {
// CHECK-NEXT:      %0, %1, %2 = "test.op"() : () -> (!riscv.reg<t0>, !riscv.reg<t1>, !riscv.reg<t2>)
// CHECK-NEXT:      %3, %4, %5 = riscv.parallel_mov %0, %1, %2 [32, 32, 32] : (!riscv.reg<t0>, !riscv.reg<t1>, !riscv.reg<t2>) -> (!riscv.reg<t0>, !riscv.reg<t1>, !riscv.reg<t2>)
// CHECK-NEXT:      "test.op"(%3, %4, %5) : (!riscv.reg<t0>, !riscv.reg<t1>, !riscv.reg<t2>) -> ()
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
