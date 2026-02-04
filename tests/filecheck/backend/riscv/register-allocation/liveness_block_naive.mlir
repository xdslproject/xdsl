// RUN: xdsl-opt -p riscv-allocate-registers{allocation_strategy=LivenessBlockNaive} %s | filecheck %s

builtin.module {
  riscv_func.func @main() {
    %0 = rv32.li 6 : !riscv.reg
    %1 = rv32.li 5 : !riscv.reg<s0>
    %2 = riscv.add %0, %1 : (!riscv.reg, !riscv.reg<s0>) -> !riscv.reg
    %3 = rv32.li 29 : !riscv.reg
    %4 = rv32.li 28 : !riscv.reg
    %5 = rv32.li 27 : !riscv.reg
    %6 = rv32.li 26 : !riscv.reg
    %7 = rv32.li 25 : !riscv.reg
    %8 = rv32.li 24 : !riscv.reg
    %9 = rv32.li 23 : !riscv.reg
    %10 = rv32.li 22 : !riscv.reg
    %11 = rv32.li 21 : !riscv.reg
    %12 = rv32.li 20 : !riscv.reg
    %13 = rv32.li 19 : !riscv.reg
    %14 = rv32.li 18 : !riscv.reg
    %15 = rv32.li 17 : !riscv.reg
    %16 = rv32.li 16 : !riscv.reg
    %17 = rv32.li 15 : !riscv.reg
    %18 = rv32.li 14 : !riscv.reg
    %19 = rv32.li 13 : !riscv.reg
    %20 = rv32.li 12 : !riscv.reg
    %21 = rv32.li 11 : !riscv.reg
    %22 = rv32.li 10 : !riscv.reg
    %23 = rv32.li 9 : !riscv.reg
    %24 = rv32.li 8 : !riscv.reg
    %25 = rv32.li 7 : !riscv.reg
    %26 = rv32.li 6 : !riscv.reg
    %27 = rv32.li 5 : !riscv.reg
    %28 = rv32.li 4 : !riscv.reg
    %29 = rv32.li 3 : !riscv.reg
    %30 = rv32.li 2 : !riscv.reg
    %31 = rv32.li 1 : !riscv.reg
    %32 = riscv.fcvt.s.w %30 : (!riscv.reg) -> !riscv.freg
    %33 = riscv.fcvt.s.w %31 : (!riscv.reg) -> !riscv.freg
    %34 = riscv.fadd.s %32, %33 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    riscv_func.return
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:  riscv_func.func @main() {
// CHECK-NEXT:    %{{\d+}} = rv32.li 6 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 5 : !riscv.reg<s0>
// CHECK-NEXT:    %{{\d+}} = riscv.add %{{\d+}}, %{{\d+}} : (!riscv.reg<t1>, !riscv.reg<s0>) -> !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 29 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 28 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 27 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 26 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 25 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 24 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 23 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 22 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 21 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 20 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 19 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 18 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 17 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 16 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 15 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 14 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 13 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 12 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 11 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 10 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 9 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 8 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 7 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 6 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 5 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 4 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 3 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 2 : !riscv.reg<t1>
// CHECK-NEXT:    %{{\d+}} = rv32.li 1 : !riscv.reg<t0>
// CHECK-NEXT:    %{{\d+}} = riscv.fcvt.s.w %{{\d+}} : (!riscv.reg<t1>) -> !riscv.freg<ft0>
// CHECK-NEXT:    %{{\d+}} = riscv.fcvt.s.w %{{\d+}} : (!riscv.reg<t0>) -> !riscv.freg<ft1>
// CHECK-NEXT:    %{{\d+}} = riscv.fadd.s %{{\d+}}, %{{\d+}} : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft0>
// CHECK-NEXT:    riscv_func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
