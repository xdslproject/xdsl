// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive}" %s | filecheck %s --check-prefix=LIVE-BNAIVE

riscv_func.func @main() {
  %0 = riscv.li 6 : () -> !riscv.reg<>
  %1 = riscv.li 5 : () -> !riscv.reg<t0>
  %3 = riscv.fcvt.s.w %0 : (!riscv.reg<>) -> !riscv.freg<>
  %4 = riscv.fcvt.s.w %1 : (!riscv.reg<t0>) -> !riscv.freg<>
  %5 = riscv.fadd.s %3, %4 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %2 = riscv.add %0, %1 : (!riscv.reg<>, !riscv.reg<t0>) -> !riscv.reg<>
  riscv_func.return
}

// LIVE-BNAIVE:       builtin.module {
// LIVE-BNAIVE-NEXT:    riscv_func.func @main() {
// LIVE-BNAIVE-NEXT:      %0 = riscv.li 6 : () -> !riscv.reg<t1>
// LIVE-BNAIVE-NEXT:      %1 = riscv.li 5 : () -> !riscv.reg<t0>
// LIVE-BNAIVE-NEXT:      %2 = riscv.fcvt.s.w %0 : (!riscv.reg<t1>) -> !riscv.freg<ft0>
// LIVE-BNAIVE-NEXT:      %3 = riscv.fcvt.s.w %1 : (!riscv.reg<t0>) -> !riscv.freg<ft1>
// LIVE-BNAIVE-NEXT:      %4 = riscv.fadd.s %2, %3 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft0>
// LIVE-BNAIVE-NEXT:      %5 = riscv.add %0, %1 : (!riscv.reg<t1>, !riscv.reg<t0>) -> !riscv.reg<t1>
// LIVE-BNAIVE-NEXT:      riscv_func.return
// LIVE-BNAIVE-NEXT:    }
// LIVE-BNAIVE-NEXT:  }
