// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive}" %s | filecheck %s --check-prefix=CHECK-LIVENESS-BLOCK-NAIVE
// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive limit_registers=0}" %s | filecheck %s --check-prefix=CHECK-LIVENESS-BLOCK-NAIVE-J

riscv_func.func @external() -> ()

riscv_func.func @main() {
  %0 = riscv.li 6 : () -> !riscv.reg<>
  %1 = riscv.li 5 : () -> !riscv.reg<s0>
  %2 = riscv.fcvt.s.w %0 : (!riscv.reg<>) -> !riscv.freg<>
  %3 = riscv.fcvt.s.w %1 : (!riscv.reg<s0>) -> !riscv.freg<>
  %4 = riscv.fadd.s %2, %3 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %5 = riscv.add %0, %1 : (!riscv.reg<>, !riscv.reg<s0>) -> !riscv.reg<>

  "riscv_scf.for"(%0, %1, %5) ({
  ^0(%6 : !riscv.reg<>):
    "riscv_scf.yield"() : () -> ()
  }) : (!riscv.reg<>, !riscv.reg<s0>, !riscv.reg<>) -> ()

  %7 = "riscv_scf.for"(%0, %1, %5, %5) ({
  ^0(%8 : !riscv.reg<>, %9 : !riscv.reg<>):
    %10 = riscv.mv %9 : (!riscv.reg<>) -> !riscv.reg<>
    "riscv_scf.yield"(%10) : (!riscv.reg<>) -> ()
  }) : (!riscv.reg<>, !riscv.reg<s0>, !riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  riscv_func.return
}

//   CHECK-LIVENESS-BLOCK-NAIVE:       builtin.module {
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:    riscv_func.func @external() -> ()
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:    riscv_func.func @main() {
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.li 6 : () -> !riscv.reg<t4>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.li 5 : () -> !riscv.reg<s0>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.fcvt.s.w %{{\d+}} : (!riscv.reg<t4>) -> !riscv.freg<ft11>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.fcvt.s.w %{{\d+}} : (!riscv.reg<s0>) -> !riscv.freg<ft10>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.fadd.s %{{\d+}}, %{{\d+}} : (!riscv.freg<ft11>, !riscv.freg<ft10>) -> !riscv.freg<ft11>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = riscv.add %{{\d+}}, %{{\d+}} : (!riscv.reg<t4>, !riscv.reg<s0>) -> !riscv.reg<t6>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      "riscv_scf.for"(%{{\d+}}, %{{\d+}}, %{{\d+}}) ({
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      ^0(%{{\d+}} : !riscv.reg<t6>):
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:        "riscv_scf.yield"() : () -> ()
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      }) : (!riscv.reg<t4>, !riscv.reg<s0>, !riscv.reg<t6>) -> ()
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      %{{\d+}} = "riscv_scf.for"(%{{\d+}}, %{{\d+}}, %{{\d+}}, %{{\d+}}) ({
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      ^1(%{{\d+}} : !riscv.reg<t5>, %{{\d+}} : !riscv.reg<t6>):
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:        %{{\d+}} = riscv.mv %9 : (!riscv.reg<t6>) -> !riscv.reg<t6>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:        "riscv_scf.yield"(%{{\d+}}) : (!riscv.reg<t6>) -> ()
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      }) : (!riscv.reg<t4>, !riscv.reg<s0>, !riscv.reg<t6>, !riscv.reg<t6>) -> !riscv.reg<t6>
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:      riscv_func.return
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:    }
//   CHECK-LIVENESS-BLOCK-NAIVE-NEXT:  }

//   CHECK-LIVENESS-BLOCK-NAIVE-J:       builtin.module {
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:    riscv_func.func @external() -> ()
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:    riscv_func.func @main() {
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %{{\d+}} = riscv.li 6 : () -> !riscv.reg<j2>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %{{\d+}} = riscv.li 5 : () -> !riscv.reg<s0>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %{{\d+}} = riscv.fcvt.s.w %{{\d+}} : (!riscv.reg<j2>) -> !riscv.freg<j3>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %{{\d+}} = riscv.fcvt.s.w %{{\d+}} : (!riscv.reg<s0>) -> !riscv.freg<j4>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %{{\d+}} = riscv.fadd.s %{{\d+}}, %{{\d+}} : (!riscv.freg<j3>, !riscv.freg<j4>) -> !riscv.freg<j3>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %{{\d+}} = riscv.add %{{\d+}}, %{{\d+}} : (!riscv.reg<j2>, !riscv.reg<s0>) -> !riscv.reg<j0>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      "riscv_scf.for"(%{{\d+}}, %{{\d+}}, %{{\d+}}) ({
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      ^0(%{{\d+}} : !riscv.reg<j0>):
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:        "riscv_scf.yield"() : () -> ()
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      }) : (!riscv.reg<j2>, !riscv.reg<s0>, !riscv.reg<j0>) -> ()
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      %{{\d+}} = "riscv_scf.for"(%{{\d+}}, %{{\d+}}, %{{\d+}}, %{{\d+}}) ({
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      ^1(%{{\d+}} : !riscv.reg<j1>, %{{\d+}} : !riscv.reg<j0>):
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:        %{{\d+}} = riscv.mv %9 : (!riscv.reg<j0>) -> !riscv.reg<j0>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:        "riscv_scf.yield"(%{{\d+}}) : (!riscv.reg<j0>) -> ()
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      }) : (!riscv.reg<j2>, !riscv.reg<s0>, !riscv.reg<j0>, !riscv.reg<j0>) -> !riscv.reg<j0>
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:      riscv_func.return
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:    }
//   CHECK-LIVENESS-BLOCK-NAIVE-J-NEXT:  }
