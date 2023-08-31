// RUN: xdsl-opt -p riscv-allocate-registers{allocation_strategy=BlockNaive} %s | filecheck %s

riscv_func.func @main() {
  %0 = riscv.li 6 : () -> !riscv.reg<>
  %1 = riscv.li 5 : () -> !riscv.reg<s0>
  %2 = riscv.add %0, %1 : (!riscv.reg<>, !riscv.reg<s0>) -> !riscv.reg<>
  %3 = riscv.li 29 : () -> !riscv.reg<>
  %4 = riscv.li 28 : () -> !riscv.reg<>
  %5 = riscv.li 27 : () -> !riscv.reg<>
  %6 = riscv.li 26 : () -> !riscv.reg<>
  %7 = riscv.li 25 : () -> !riscv.reg<>
  %8 = riscv.li 24 : () -> !riscv.reg<>
  %9 = riscv.li 23 : () -> !riscv.reg<>
  %10 = riscv.li 22 : () -> !riscv.reg<>
  %11 = riscv.li 21 : () -> !riscv.reg<>
  %12 = riscv.li 20 : () -> !riscv.reg<>
  %13 = riscv.li 19 : () -> !riscv.reg<>
  %14 = riscv.li 18 : () -> !riscv.reg<>
  %15 = riscv.li 17 : () -> !riscv.reg<>
  %16 = riscv.li 16 : () -> !riscv.reg<>
  %17 = riscv.li 15 : () -> !riscv.reg<>
  %18 = riscv.li 14 : () -> !riscv.reg<>
  %19 = riscv.li 13 : () -> !riscv.reg<>
  %20 = riscv.li 12 : () -> !riscv.reg<>
  %21 = riscv.li 11 : () -> !riscv.reg<>
  %22 = riscv.li 10 : () -> !riscv.reg<>
  %23 = riscv.li 9 : () -> !riscv.reg<>
  %24 = riscv.li 8 : () -> !riscv.reg<>
  %25 = riscv.li 7 : () -> !riscv.reg<>
  %26 = riscv.li 6 : () -> !riscv.reg<>
  %27 = riscv.li 5 : () -> !riscv.reg<>
  %28 = riscv.li 4 : () -> !riscv.reg<>
  %29 = riscv.li 3 : () -> !riscv.reg<>
  %30 = riscv.li 2 : () -> !riscv.reg<>
  %31 = riscv.li 1 : () -> !riscv.reg<>
  %32 = riscv.fcvt.s.w %30 : (!riscv.reg<>) -> !riscv.freg<>
  %33 = riscv.fcvt.s.w %31 : (!riscv.reg<>) -> !riscv.freg<>
  %34 = riscv.fadd.s %32, %33 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  riscv_func.return
}

// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() {
// CHECK-NEXT:      %{{\d+}} = riscv.li 6 : () -> !riscv.reg<t6>
// CHECK-NEXT:      %{{\d+}} = riscv.li 5 : () -> !riscv.reg<s0>
// CHECK-NEXT:      %{{\d+}} = riscv.add %{{\d+}}, %{{\d+}} : (!riscv.reg<t6>, !riscv.reg<s0>) -> !riscv.reg<t5>
// CHECK-NEXT:      %{{\d+}} = riscv.li 29 : () -> !riscv.reg<t4>
// CHECK-NEXT:      %{{\d+}} = riscv.li 28 : () -> !riscv.reg<t3>
// CHECK-NEXT:      %{{\d+}} = riscv.li 27 : () -> !riscv.reg<s11>
// CHECK-NEXT:      %{{\d+}} = riscv.li 26 : () -> !riscv.reg<s10>
// CHECK-NEXT:      %{{\d+}} = riscv.li 25 : () -> !riscv.reg<s9>
// CHECK-NEXT:      %{{\d+}} = riscv.li 24 : () -> !riscv.reg<s8>
// CHECK-NEXT:      %{{\d+}} = riscv.li 23 : () -> !riscv.reg<s7>
// CHECK-NEXT:      %{{\d+}} = riscv.li 22 : () -> !riscv.reg<s6>
// CHECK-NEXT:      %{{\d+}} = riscv.li 21 : () -> !riscv.reg<s5>
// CHECK-NEXT:      %{{\d+}} = riscv.li 20 : () -> !riscv.reg<s4>
// CHECK-NEXT:      %{{\d+}} = riscv.li 19 : () -> !riscv.reg<s3>
// CHECK-NEXT:      %{{\d+}} = riscv.li 18 : () -> !riscv.reg<s2>
// CHECK-NEXT:      %{{\d+}} = riscv.li 17 : () -> !riscv.reg<a7>
// CHECK-NEXT:      %{{\d+}} = riscv.li 16 : () -> !riscv.reg<a6>
// CHECK-NEXT:      %{{\d+}} = riscv.li 15 : () -> !riscv.reg<a5>
// CHECK-NEXT:      %{{\d+}} = riscv.li 14 : () -> !riscv.reg<a4>
// CHECK-NEXT:      %{{\d+}} = riscv.li 13 : () -> !riscv.reg<a3>
// CHECK-NEXT:      %{{\d+}} = riscv.li 12 : () -> !riscv.reg<a2>
// CHECK-NEXT:      %{{\d+}} = riscv.li 11 : () -> !riscv.reg<a1>
// CHECK-NEXT:      %{{\d+}} = riscv.li 10 : () -> !riscv.reg<a0>
// CHECK-NEXT:      %{{\d+}} = riscv.li 9 : () -> !riscv.reg<s1>
// CHECK-NEXT:      %{{\d+}} = riscv.li 8 : () -> !riscv.reg<t2>
// CHECK-NEXT:      %{{\d+}} = riscv.li 7 : () -> !riscv.reg<t1>
// CHECK-NEXT:      %{{\d+}} = riscv.li 6 : () -> !riscv.reg<t0>
// CHECK-NEXT:      %{{\d+}} = riscv.li 5 : () -> !riscv.reg<ra>
// CHECK-NEXT:      %{{\d+}} = riscv.li 4 : () -> !riscv.reg<j0>
// CHECK-NEXT:      %{{\d+}} = riscv.li 3 : () -> !riscv.reg<j1>
// CHECK-NEXT:      %{{\d+}} = riscv.li 2 : () -> !riscv.reg<j2>
// CHECK-NEXT:      %{{\d+}} = riscv.li 1 : () -> !riscv.reg<j3>
// CHECK-NEXT:      %{{\d+}} = riscv.fcvt.s.w %{{\d+}} : (!riscv.reg<j2>) -> !riscv.freg<ft11>
// CHECK-NEXT:      %{{\d+}} = riscv.fcvt.s.w %{{\d+}} : (!riscv.reg<j3>) -> !riscv.freg<ft10>
// CHECK-NEXT:      %{{\d+}} = riscv.fadd.s %{{\d+}}, %{{\d+}} : (!riscv.freg<ft11>, !riscv.freg<ft10>) -> !riscv.freg<ft9>
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
