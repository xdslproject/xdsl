// RUN: xdsl-opt -p riscv-allocate-registers{allocation_strategy=BlockNaive} %s --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<x$>
  %1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<s0>
  %2 = "riscv.add"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<s0>) -> !riscv.reg<x$>
  %3 = "riscv.li"() {"immediate" = 29 : i32} : () -> !riscv.reg<x$>
  %4 = "riscv.li"() {"immediate" = 28 : i32} : () -> !riscv.reg<x$>
  %5 = "riscv.li"() {"immediate" = 27 : i32} : () -> !riscv.reg<x$>
  %6 = "riscv.li"() {"immediate" = 26 : i32} : () -> !riscv.reg<x$>
  %7 = "riscv.li"() {"immediate" = 25 : i32} : () -> !riscv.reg<x$>
  %8 = "riscv.li"() {"immediate" = 24 : i32} : () -> !riscv.reg<x$>
  %9 = "riscv.li"() {"immediate" = 23 : i32} : () -> !riscv.reg<x$>
  %10 = "riscv.li"() {"immediate" = 22 : i32} : () -> !riscv.reg<x$>
  %11 = "riscv.li"() {"immediate" = 21 : i32} : () -> !riscv.reg<x$>
  %12 = "riscv.li"() {"immediate" = 20 : i32} : () -> !riscv.reg<x$>
  %13 = "riscv.li"() {"immediate" = 19 : i32} : () -> !riscv.reg<x$>
  %14 = "riscv.li"() {"immediate" = 18 : i32} : () -> !riscv.reg<x$>
  %15 = "riscv.li"() {"immediate" = 17 : i32} : () -> !riscv.reg<x$>
  %16 = "riscv.li"() {"immediate" = 16 : i32} : () -> !riscv.reg<x$>
  %17 = "riscv.li"() {"immediate" = 15 : i32} : () -> !riscv.reg<x$>
  %18 = "riscv.li"() {"immediate" = 14 : i32} : () -> !riscv.reg<x$>
  %19 = "riscv.li"() {"immediate" = 13 : i32} : () -> !riscv.reg<x$>
  %20 = "riscv.li"() {"immediate" = 12 : i32} : () -> !riscv.reg<x$>
  %21 = "riscv.li"() {"immediate" = 11 : i32} : () -> !riscv.reg<x$>
  %22 = "riscv.li"() {"immediate" = 10 : i32} : () -> !riscv.reg<x$>
  %23 = "riscv.li"() {"immediate" = 9 : i32} : () -> !riscv.reg<x$>
  %24 = "riscv.li"() {"immediate" = 8 : i32} : () -> !riscv.reg<x$>
  %25 = "riscv.li"() {"immediate" = 7 : i32} : () -> !riscv.reg<x$>
  %26 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<x$>
  %27 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<x$>
  %28 = "riscv.li"() {"immediate" = 4 : i32} : () -> !riscv.reg<x$>
  %29 = "riscv.li"() {"immediate" = 3 : i32} : () -> !riscv.reg<x$>
  %30 = "riscv.li"() {"immediate" = 2 : i32} : () -> !riscv.reg<x$>
  %31 = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<x$>
  %32 = "riscv.fcvt.s.w"(%30) : (!riscv.reg<x$>) -> !riscv.freg<f$>
  %33 = "riscv.fcvt.s.w"(%31) : (!riscv.reg<x$>) -> !riscv.freg<f$>
  %34 = "riscv.fadd.s"(%32, %33) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<t6>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<s0>
// CHECK-NEXT:  %{{\d+}} = "riscv.add"(%{{\d+}}, %{{\d+}}) : (!riscv.reg<t6>, !riscv.reg<s0>) -> !riscv.reg<t5>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 29 : i32} : () -> !riscv.reg<t4>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 28 : i32} : () -> !riscv.reg<t3>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 27 : i32} : () -> !riscv.reg<s11>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 26 : i32} : () -> !riscv.reg<s10>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 25 : i32} : () -> !riscv.reg<s9>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 24 : i32} : () -> !riscv.reg<s8>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 23 : i32} : () -> !riscv.reg<s7>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 22 : i32} : () -> !riscv.reg<s6>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 21 : i32} : () -> !riscv.reg<s5>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 20 : i32} : () -> !riscv.reg<s4>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 19 : i32} : () -> !riscv.reg<s3>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 18 : i32} : () -> !riscv.reg<s2>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 17 : i32} : () -> !riscv.reg<a7>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 16 : i32} : () -> !riscv.reg<a6>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 15 : i32} : () -> !riscv.reg<a5>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 14 : i32} : () -> !riscv.reg<a4>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 13 : i32} : () -> !riscv.reg<a3>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 12 : i32} : () -> !riscv.reg<a2>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 11 : i32} : () -> !riscv.reg<a1>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 10 : i32} : () -> !riscv.reg<a0>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 9 : i32} : () -> !riscv.reg<s1>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 8 : i32} : () -> !riscv.reg<t2>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 7 : i32} : () -> !riscv.reg<t1>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<t0>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<ra>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 4 : i32} : () -> !riscv.reg<j0>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 3 : i32} : () -> !riscv.reg<j1>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 2 : i32} : () -> !riscv.reg<j2>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<j3>
// CHECK-NEXT:  %{{\d+}} = "riscv.fcvt.s.w"(%{{\d+}}) : (!riscv.reg<j2>) -> !riscv.freg<ft11>
// CHECK-NEXT:  %{{\d+}} = "riscv.fcvt.s.w"(%{{\d+}}) : (!riscv.reg<j3>) -> !riscv.freg<ft10>
// CHECK-NEXT:  %{{\d+}} = "riscv.fadd.s"(%{{\d+}}, %{{\d+}}) : (!riscv.freg<ft11>, !riscv.freg<ft10>) -> !riscv.freg<ft9>
// CHECK-NEXT: }) : () -> ()
