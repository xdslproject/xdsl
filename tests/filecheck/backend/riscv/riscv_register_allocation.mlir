// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive}" "%s" | filecheck "%s" --check-prefix=CHECK-LIVENESS-BLOCK-NAIVE
// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive limit_registers=0}" "%s" | filecheck "%s" --check-prefix=CHECK-LIVENESS-BLOCK-NAIVE-J

riscv_func.func @external() -> ()

riscv_func.func @main() {
  %zero = riscv.li 0 : () -> !riscv.reg<>
  %0 = riscv.li 6 : () -> !riscv.reg<>
  %1 = riscv.li 5 : () -> !riscv.reg<s0>
  %2 = riscv.fcvt.s.w %0 : (!riscv.reg<>) -> !riscv.freg<>
  %3 = riscv.fcvt.s.w %1 : (!riscv.reg<s0>) -> !riscv.freg<>
  %4 = riscv.fadd.s %2, %3 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %5 = riscv.add %0, %1 : (!riscv.reg<>, !riscv.reg<s0>) -> !riscv.reg<>

  riscv_scf.for %6 : !riscv.reg<> = %0 to %1 step %5 {
  }

  %7 = riscv_scf.for %8 : !riscv.reg<> = %0 to  %1 step %5 iter_args(%9 = %5) -> (!riscv.reg<>) {
    %10 = riscv.mv %9 : (!riscv.reg<>) -> !riscv.reg<>
    riscv_scf.yield %10 : !riscv.reg<>
  }

  %zero_0 = riscv.li 0 : () -> !riscv.reg<>
  %zero_1 = riscv.li 0 : () -> !riscv.reg<a0>

  riscv_func.return
}

//   CHECK-LIVENESS-BLOCK-NAIVE:       builtin.module {
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:    riscv_func.func @external() -> ()
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:    riscv_func.func @main() {
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{.+}} = riscv.li 0 : () -> !riscv.reg<zero>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.li 6 : () -> !riscv.reg<t2>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.li 5 : () -> !riscv.reg<s0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.fcvt.s.w %{{\d+}} : (!riscv.reg<t2>) -> !riscv.freg<ft0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.fcvt.s.w %{{\d+}} : (!riscv.reg<s0>) -> !riscv.freg<ft1>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.fadd.s %{{\d+}}, %{{\d+}} : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.add %{{\d+}}, %{{\d+}} : (!riscv.reg<t2>, !riscv.reg<s0>) -> !riscv.reg<t0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      riscv_scf.for %{{\d+}} : !riscv.reg<t0> = %{{\d+}} to %{{\d+}} step %{{\d+}} {
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      }
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv_scf.for %{{\d+}} : !riscv.reg<t1> = %{{\d+}} to %{{\d+}} step %{{\d+}} iter_args(%{{\d+}} = %{{\d+}}) -> (!riscv.reg<t0>) {
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:        %{{\d+}} = riscv.mv %9 : (!riscv.reg<t0>) -> !riscv.reg<t0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:        riscv_scf.yield %{{\d+}} : !riscv.reg<t0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      }
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %zero_1 = riscv.li 0 : () -> !riscv.reg<zero>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %zero_2 = riscv.li 0 : () -> !riscv.reg<a0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      riscv_func.return
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:    }
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:  }

//   CHECK-LIVENESS-BLOCK-NAIVE-J:       builtin.module {
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:    riscv_func.func @external() -> ()
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:    riscv_func.func @main() {
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %{{.+}} = riscv.li 0 : () -> !riscv.reg<zero>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %{{\d+}} = riscv.li 6 : () -> !riscv.reg<j2>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %{{\d+}} = riscv.li 5 : () -> !riscv.reg<s0>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %{{\d+}} = riscv.fcvt.s.w %{{\d+}} : (!riscv.reg<j2>) -> !riscv.freg<j3>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %{{\d+}} = riscv.fcvt.s.w %{{\d+}} : (!riscv.reg<s0>) -> !riscv.freg<j4>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %{{\d+}} = riscv.fadd.s %{{\d+}}, %{{\d+}} : (!riscv.freg<j3>, !riscv.freg<j4>) -> !riscv.freg<j3>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %{{\d+}} = riscv.add %{{\d+}}, %{{\d+}} : (!riscv.reg<j2>, !riscv.reg<s0>) -> !riscv.reg<j0>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      riscv_scf.for %{{\d+}} : !riscv.reg<j0> = %{{\d+}} to %{{\d+}} step %{{\d+}} {
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      }
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %{{\d+}} = riscv_scf.for %{{\d+}} : !riscv.reg<j1> = %{{\d+}} to %{{\d+}} step %{{\d+}} iter_args(%{{\d+}} = %{{\d+}}) -> (!riscv.reg<j0>) {
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:        %{{\d+}} = riscv.mv %9 : (!riscv.reg<j0>) -> !riscv.reg<j0>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:        riscv_scf.yield %{{\d+}} : !riscv.reg<j0>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      }
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %zero_1 = riscv.li 0 : () -> !riscv.reg<zero>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %zero_2 = riscv.li 0 : () -> !riscv.reg<a0>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      riscv_func.return
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:    }
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:  }
