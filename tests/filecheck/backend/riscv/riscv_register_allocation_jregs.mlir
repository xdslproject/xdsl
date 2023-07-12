// RUN: xdsl-opt -p riscv-allocate-registers{allocation_strategy=GlobalJRegs} %s --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<x$>
  %1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<s0>
  %3 = "riscv.fcvt.s.w"(%0) : (!riscv.reg<x$>) -> !riscv.freg<f$>
  %4 = "riscv.fcvt.s.w"(%1) : (!riscv.reg<s0>) -> !riscv.freg<f$>
  %5 = "riscv.fadd.s"(%3, %4) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %2 = "riscv.add"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<s0>) -> !riscv.reg<x$>
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<j0>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<s0>
// CHECK-NEXT:  %{{\d+}} = "riscv.fcvt.s.w"(%{{\d+}}) : (!riscv.reg<j0>) -> !riscv.freg<j1>
// CHECK-NEXT:  %{{\d+}} = "riscv.fcvt.s.w"(%{{\d+}}) : (!riscv.reg<s0>) -> !riscv.freg<j2>
// CHECK-NEXT:  %{{\d+}} = "riscv.fadd.s"(%{{\d+}}, %{{\d+}}) : (!riscv.freg<j1>, !riscv.freg<j2>) -> !riscv.freg<j3>
// CHECK-NEXT:  %{{\d+}} = "riscv.add"(%{{\d+}}, %{{\d+}}) : (!riscv.reg<j0>, !riscv.reg<s0>) -> !riscv.reg<j4>
// CHECK-NEXT: }) : () -> ()
