// RUN: xdsl-opt -p riscv-reorder-infinite --split-input-file --verify-diagnostics %s | filecheck %s
// Do nothing case:
builtin.module {
  riscv_func.func @main() -> (!riscv.freg, !riscv.freg) {
    %0, %1, %2 = "test.op"() : () -> (!riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>)
    %3, %4, %5 = riscv.parallel_mov %0, %1, %2 [32, 32, 32] : (!riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>) -> (!riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>)
    riscv_func.return %3, %4, %5 : !riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>
  }
}
// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() -> (!riscv.freg, !riscv.freg) {
// CHECK-NEXT:      %0, %1, %2 = "test.op"() : () -> (!riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>)
// CHECK-NEXT:      %3, %4, %5 = riscv.parallel_mov %0, %1, %2 [32, 32, 32] : (!riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>) -> (!riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>)
// CHECK-NEXT:      riscv_func.return %3, %4, %5 : !riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----
// Swap with preallocated
builtin.module {
  riscv_func.func @main() -> (!riscv.freg, !riscv.freg) {
    %0, %1, %2 = "test.op"() : () -> (!riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>)
    %3, %4, %5 = riscv.parallel_mov %0, %1, %2 [32, 32, 32] : (!riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>) -> (!riscv.freg<ft0>, !riscv.freg<fj_0>, !riscv.freg<fj_1>)
    riscv_func.return %3, %4, %5 : !riscv.freg<ft0>, !riscv.freg<fj_0>, !riscv.freg<fj_1>
  }
}
// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() -> (!riscv.freg, !riscv.freg) {
// CHECK-NEXT:      %0, %1, %2 = "test.op"() : () -> (!riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>)
// CHECK-NEXT:      %3, %4, %5 = riscv.parallel_mov %0, %1, %2 [32, 32, 32] : (!riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>) -> (!riscv.freg<ft0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>)
// CHECK-NEXT:      riscv_func.return %3, %4, %5 : !riscv.freg<ft0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
