// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive}" %s | filecheck %s --check-prefix=CHECK-LIVENESS-BLOCK-NAIVE

riscv_func.func @main() {
  %0 = riscv.li 6 : !riscv.reg
  %1 = riscv.li 5 : !riscv.reg<s0>
  %2 = riscv.fcvt.s.w %0 : (!riscv.reg) -> !riscv.freg
  %3 = riscv.fcvt.s.w %1 : (!riscv.reg<s0>) -> !riscv.freg
  %4 = riscv.fadd.s %2, %3 : (!riscv.freg, !riscv.freg) -> !riscv.freg
  %5 = riscv.add %0, %1 : (!riscv.reg, !riscv.reg<s0>) -> !riscv.reg

  riscv_snitch.frep_outer %0 {
  }

  %7 = riscv_snitch.frep_outer %0 iter_args(%6 = %5) -> (!riscv.reg) {
    %8 = riscv.mv %6 : (!riscv.reg) -> !riscv.reg
    riscv_snitch.frep_yield %8 : !riscv.reg
  }
  riscv_func.return
}

//   CHECK-LIVENESS-BLOCK-NAIVE:       builtin.module {
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:    riscv_func.func @main() {
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.li 6 : !riscv.reg<t1>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.li 5 : !riscv.reg<s0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.fcvt.s.w %{{\d+}} : (!riscv.reg<t1>) -> !riscv.freg<ft0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.fcvt.s.w %{{\d+}} : (!riscv.reg<s0>) -> !riscv.freg<ft1>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.fadd.s %{{\d+}}, %{{\d+}} : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.add %{{\d+}}, %{{\d+}} : (!riscv.reg<t1>, !riscv.reg<s0>) -> !riscv.reg<t0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      riscv_snitch.frep_outer %{{\d+}} {
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      }
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv_snitch.frep_outer %{{\d+}} iter_args(%{{\d+}} = %{{\d+}}) -> (!riscv.reg<t0>) {
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:        %{{\d+}} = riscv.mv %{{\d+}} : (!riscv.reg<t0>) -> !riscv.reg<t0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:        riscv_snitch.frep_yield %{{\d+}} : !riscv.reg<t0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      }
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      riscv_func.return
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:    }
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:  }
