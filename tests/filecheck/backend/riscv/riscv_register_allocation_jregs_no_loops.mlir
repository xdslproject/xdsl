// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=BlockNaive limit_registers=0}" %s | filecheck %s

riscv_func.func @main() {
  %0 = riscv.li 6 : () -> !riscv.reg<>
  %1 = riscv.li 5 : () -> !riscv.reg<s0>
  %2 = riscv.fcvt.s.w %0 : (!riscv.reg<>) -> !riscv.freg<>
  %3 = riscv.fcvt.s.w %1 : (!riscv.reg<s0>) -> !riscv.freg<>
  %4 = riscv.fadd.s %2, %3 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %5 = riscv.add %0, %1 : (!riscv.reg<>, !riscv.reg<s0>) -> !riscv.reg<>
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
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
