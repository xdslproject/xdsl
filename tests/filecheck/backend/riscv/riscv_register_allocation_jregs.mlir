// RUN: xdsl-opt -p riscv-allocate-registers{allocation_strategy=GlobalJRegs} %s --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.ireg<>
  %1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.ireg<s0>
  %3 = "riscv.fcvt.s.w"(%0) : (!riscv.ireg<>) -> !riscv.freg<>
  %4 = "riscv.fcvt.s.w"(%1) : (!riscv.ireg<s0>) -> !riscv.freg<>
  %5 = "riscv.fadd.s"(%3, %4) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %2 = "riscv.add"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<s0>) -> !riscv.ireg<>
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.ireg<j0>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.ireg<s0>
// CHECK-NEXT:  %{{\d+}} = "riscv.fcvt.s.w"(%{{\d+}}) : (!riscv.ireg<j0>) -> !riscv.freg<j1>
// CHECK-NEXT:  %{{\d+}} = "riscv.fcvt.s.w"(%{{\d+}}) : (!riscv.ireg<s0>) -> !riscv.freg<j2>
// CHECK-NEXT:  %{{\d+}} = "riscv.fadd.s"(%{{\d+}}, %{{\d+}}) : (!riscv.freg<j1>, !riscv.freg<j2>) -> !riscv.freg<j3>
// CHECK-NEXT:  %{{\d+}} = "riscv.add"(%{{\d+}}, %{{\d+}}) : (!riscv.ireg<j0>, !riscv.ireg<s0>) -> !riscv.ireg<j4>
// CHECK-NEXT: }) : () -> ()
