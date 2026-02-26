// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive}" %s | filecheck %s --check-prefix=CHECK-LIVENESS-BLOCK-NAIVE

riscv_func.func @external() -> ()

riscv_func.func @main() {
  %zero = rv32.li 0 : !riscv.reg
  %0 = rv32.li 6 : !riscv.reg
  %1 = rv32.li 5 : !riscv.reg<s0>
  %2 = riscv.fcvt.s.w %0 : (!riscv.reg) -> !riscv.freg
  %3 = riscv.fcvt.s.w %1 : (!riscv.reg<s0>) -> !riscv.freg
  %4 = riscv.fadd.s %2, %3 : (!riscv.freg, !riscv.freg) -> !riscv.freg
  %5 = riscv.add %0, %1 : (!riscv.reg, !riscv.reg<s0>) -> !riscv.reg

  riscv_scf.for %6 : !riscv.reg = %0 to %1 step %5 {
  }

  %7 = riscv_scf.for %8 : !riscv.reg = %0 to  %1 step %5 iter_args(%9 = %5) -> (!riscv.reg) {
    %10 = riscv.mv %9 : (!riscv.reg) -> !riscv.reg
    riscv_scf.yield %10 : !riscv.reg
  }

  %zero_0 = rv32.li 0 : !riscv.reg
  %zero_1 = rv32.li 0 : !riscv.reg<a0>

  riscv_func.return
}

//   CHECK-LIVENESS-BLOCK-NAIVE:       builtin.module {
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:    riscv_func.func @external() -> ()
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:    riscv_func.func @main() {
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %zero = rv32.li 0 : !riscv.reg<zero>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %0 = rv32.li 6 : !riscv.reg<t1>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %1 = rv32.li 5 : !riscv.reg<s0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %2 = riscv.fcvt.s.w %0 : (!riscv.reg<t1>) -> !riscv.freg<ft0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %3 = riscv.fcvt.s.w %1 : (!riscv.reg<s0>) -> !riscv.freg<ft1>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %4 = riscv.fadd.s %2, %3 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %5 = riscv.add %0, %1 : (!riscv.reg<t1>, !riscv.reg<s0>) -> !riscv.reg<t0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      riscv_scf.for %6 : !riscv.reg<t2> = %0 to %1 step %5 {
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      }
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %7 = riscv_scf.for %8 : !riscv.reg<t1> = %0 to %1 step %5 iter_args(%9 = %5) -> (!riscv.reg<t0>) {
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:        %10 = riscv.mv %9 : (!riscv.reg<t0>) -> !riscv.reg<t0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:        riscv_scf.yield %10 : !riscv.reg<t0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      }
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %zero_1 = rv32.li 0 : !riscv.reg<zero>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %zero_2 = rv32.li 0 : !riscv.reg<a0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      riscv_func.return
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:    }
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:  }
