// RUN: xdsl-opt -p riscv-allocate-registers %s | filecheck %s

"builtin.module"() ({
  %0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<>
  %1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<s0>
  %2 = "riscv.add"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<j0>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<s0>
// CHECK-NEXT:  %{{\d+}} = "riscv.add"(%{{\d+}}, %{{\d+}}) : (!riscv.reg<j0>, !riscv.reg<s0>) -> !riscv.reg<j1>
// CHECK-NEXT: }) : () -> ()
