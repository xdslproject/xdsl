// RUN: xdsl-opt -p riscv-allocate-infinite-registers --split-input-file --verify-diagnostics %s | filecheck %s

builtin.module {
  riscv_func.func @main() {
    %zero = riscv.li 0 : !riscv.reg<zero>
    // j_1 shouldn't be allocated to t0
    %0 = riscv.li 6 : !riscv.reg<j_1>
    %1 = riscv.li 5 : !riscv.reg<t0>
    %2 = riscv.add %0, %1 : (!riscv.reg<j_1>, !riscv.reg<t0>) -> !riscv.reg<j_0>
    // floats work
    %3 = riscv.fcvt.s.w %1 : (!riscv.reg<t0>) -> !riscv.freg<fj_0>
    %4 = riscv.fcvt.s.w %0 : (!riscv.reg<j_1>) -> !riscv.freg<fj_1>
    // reg with multiple results (fj_1)
    %5 = riscv.fadd.s %3, %4 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_1>
    riscv_func.return
  }
}
// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() {
// CHECK-NEXT:      %zero = riscv.li 0 : !riscv.reg<zero>
// CHECK-NEXT:      %0 = riscv.li 6 : !riscv.reg<t1>
// CHECK-NEXT:      %1 = riscv.li 5 : !riscv.reg<t0>
// CHECK-NEXT:      %2 = riscv.add %0, %1 : (!riscv.reg<t1>, !riscv.reg<t0>) -> !riscv.reg<t2>
// CHECK-NEXT:      %3 = riscv.fcvt.s.w %1 : (!riscv.reg<t0>) -> !riscv.freg<ft0>
// CHECK-NEXT:      %4 = riscv.fcvt.s.w %0 : (!riscv.reg<t1>) -> !riscv.freg<ft1>
// CHECK-NEXT:      %5 = riscv.fadd.s %3, %4 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft1>
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
