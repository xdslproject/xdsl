// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive limit_registers=2}" %s | filecheck %s

riscv_func.func @main() {
  %0 = riscv.li 6 : !riscv.reg
  %1 = riscv.li 5 : !riscv.reg<s0>
  %2 = riscv.add %0, %1 : (!riscv.reg, !riscv.reg<s0>) -> !riscv.reg
  %3 = riscv.li 29 : !riscv.reg
  %4 = riscv.li 28 : !riscv.reg
  %5 = riscv.add %3, %4 : (!riscv.reg, !riscv.reg) -> !riscv.reg
  %6 = riscv.li 26 : !riscv.reg
  %7 = riscv.li 25 : !riscv.reg
  %8 = riscv.li 24 : !riscv.reg
  %9 = riscv.li 23 : !riscv.reg
  %10 = riscv.li 2 : !riscv.reg
  %11 = riscv.li 1 : !riscv.reg
  %12 = riscv.fcvt.s.w %10 : (!riscv.reg) -> !riscv.freg
  %13 = riscv.fcvt.s.w %11 : (!riscv.reg) -> !riscv.freg
  %14 = riscv.fadd.s %12, %13 : (!riscv.freg, !riscv.freg) -> !riscv.freg
  riscv_func.return
}
// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() {
// CHECK-NEXT:      %0 = riscv.li 6 : !riscv.reg<t1>
// CHECK-NEXT:      %1 = riscv.li 5 : !riscv.reg<s0>
// CHECK-NEXT:      %2 = riscv.add %0, %1 : (!riscv.reg<t1>, !riscv.reg<s0>) -> !riscv.reg<t1>
// CHECK-NEXT:      %3 = riscv.li 29 : !riscv.reg<t1>
// CHECK-NEXT:      %4 = riscv.li 28 : !riscv.reg<t0>
// CHECK-NEXT:      %5 = riscv.add %3, %4 : (!riscv.reg<t1>, !riscv.reg<t0>) -> !riscv.reg<t1>
// CHECK-NEXT:      %6 = riscv.li 26 : !riscv.reg<t1>
// CHECK-NEXT:      %7 = riscv.li 25 : !riscv.reg<t1>
// CHECK-NEXT:      %8 = riscv.li 24 : !riscv.reg<t1>
// CHECK-NEXT:      %9 = riscv.li 23 : !riscv.reg<t1>
// CHECK-NEXT:      %10 = riscv.li 2 : !riscv.reg<t1>
// CHECK-NEXT:      %11 = riscv.li 1 : !riscv.reg<t0>
// CHECK-NEXT:      %12 = riscv.fcvt.s.w %10 : (!riscv.reg<t1>) -> !riscv.freg<ft0>
// CHECK-NEXT:      %13 = riscv.fcvt.s.w %11 : (!riscv.reg<t0>) -> !riscv.freg<ft1>
// CHECK-NEXT:      %14 = riscv.fadd.s %12, %13 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft0>
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
