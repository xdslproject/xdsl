// RUN: xdsl-opt -p riscv-allocate-registers{allocation_strategy=GlobalJRegs} %s | filecheck %s

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

// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() {
// CHECK-NEXT:      %{{\d+}} = riscv.li 6 : () -> !riscv.reg<j0>
// CHECK-NEXT:      %{{\d+}} = riscv.li 5 : () -> !riscv.reg<s0>
// CHECK-NEXT:      %{{\d+}} = riscv.fcvt.s.w %{{\d+}} : (!riscv.reg<j0>) -> !riscv.freg<j1>
// CHECK-NEXT:      %{{\d+}} = riscv.fcvt.s.w %{{\d+}} : (!riscv.reg<s0>) -> !riscv.freg<j2>
// CHECK-NEXT:      %{{\d+}} = riscv.fadd.s %{{\d+}}, %{{\d+}} : (!riscv.freg<j1>, !riscv.freg<j2>) -> !riscv.freg<j3>
// CHECK-NEXT:      %{{\d+}} = riscv.add %{{\d+}}, %{{\d+}} : (!riscv.reg<j0>, !riscv.reg<s0>) -> !riscv.reg<j4>
// CHECK-NEXT:      "riscv_scf.for"(%{{\d+}}, %{{\d+}}, %{{\d+}}) ({
// CHECK-NEXT:      ^0(%{{\d+}} : !riscv.reg<j5>):
// CHECK-NEXT:        "riscv_scf.yield"() : () -> ()
// CHECK-NEXT:      }) : (!riscv.reg<j0>, !riscv.reg<s0>, !riscv.reg<j4>) -> ()
// CHECK-NEXT:      %{{\d+}} = "riscv_scf.for"(%{{\d+}}, %{{\d+}}, %{{\d+}}, %{{\d+}}) ({
// CHECK-NEXT:      ^1(%{{\d+}} : !riscv.reg<j6>, %{{\d+}} : !riscv.reg<j4>):
// CHECK-NEXT:        %{{\d+}} = riscv.mv %9 : (!riscv.reg<j4>) -> !riscv.reg<j4>
// CHECK-NEXT:        "riscv_scf.yield"(%{{\d+}}) : (!riscv.reg<j4>) -> ()
// CHECK-NEXT:      }) : (!riscv.reg<j0>, !riscv.reg<s0>, !riscv.reg<j4>, !riscv.reg<j4>) -> !riscv.reg<j4>
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
