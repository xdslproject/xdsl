// RUN: xdsl-opt -p riscv-allocate-registers{allocation_strategy=BlockNaive} %s --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.ireg<>
  %1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.ireg<s0>
  %2 = "riscv.add"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<s0>) -> !riscv.ireg<>
  %3 = "riscv.li"() {"immediate" = 29 : i32} : () -> !riscv.ireg<>
  %4 = "riscv.li"() {"immediate" = 28 : i32} : () -> !riscv.ireg<>
  %5 = "riscv.li"() {"immediate" = 27 : i32} : () -> !riscv.ireg<>
  %6 = "riscv.li"() {"immediate" = 26 : i32} : () -> !riscv.ireg<>
  %7 = "riscv.li"() {"immediate" = 25 : i32} : () -> !riscv.ireg<>
  %8 = "riscv.li"() {"immediate" = 24 : i32} : () -> !riscv.ireg<>
  %9 = "riscv.li"() {"immediate" = 23 : i32} : () -> !riscv.ireg<>
  %10 = "riscv.li"() {"immediate" = 22 : i32} : () -> !riscv.ireg<>
  %11 = "riscv.li"() {"immediate" = 21 : i32} : () -> !riscv.ireg<>
  %12 = "riscv.li"() {"immediate" = 20 : i32} : () -> !riscv.ireg<>
  %13 = "riscv.li"() {"immediate" = 19 : i32} : () -> !riscv.ireg<>
  %14 = "riscv.li"() {"immediate" = 18 : i32} : () -> !riscv.ireg<>
  %15 = "riscv.li"() {"immediate" = 17 : i32} : () -> !riscv.ireg<>
  %16 = "riscv.li"() {"immediate" = 16 : i32} : () -> !riscv.ireg<>
  %17 = "riscv.li"() {"immediate" = 15 : i32} : () -> !riscv.ireg<>
  %18 = "riscv.li"() {"immediate" = 14 : i32} : () -> !riscv.ireg<>
  %19 = "riscv.li"() {"immediate" = 13 : i32} : () -> !riscv.ireg<>
  %20 = "riscv.li"() {"immediate" = 12 : i32} : () -> !riscv.ireg<>
  %21 = "riscv.li"() {"immediate" = 11 : i32} : () -> !riscv.ireg<>
  %22 = "riscv.li"() {"immediate" = 10 : i32} : () -> !riscv.ireg<>
  %23 = "riscv.li"() {"immediate" = 9 : i32} : () -> !riscv.ireg<>
  %24 = "riscv.li"() {"immediate" = 8 : i32} : () -> !riscv.ireg<>
  %25 = "riscv.li"() {"immediate" = 7 : i32} : () -> !riscv.ireg<>
  %26 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.ireg<>
  %27 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.ireg<>
  %28 = "riscv.li"() {"immediate" = 4 : i32} : () -> !riscv.ireg<>
  %29 = "riscv.li"() {"immediate" = 3 : i32} : () -> !riscv.ireg<>
  %30 = "riscv.li"() {"immediate" = 2 : i32} : () -> !riscv.ireg<>
  %31 = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.ireg<>
  %32 = "riscv.fcvt.s.w"(%30) : (!riscv.ireg<>) -> !riscv.freg<>
  %33 = "riscv.fcvt.s.w"(%31) : (!riscv.ireg<>) -> !riscv.freg<>
  %34 = "riscv.fadd.s"(%32, %33) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.ireg<t6>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.ireg<s0>
// CHECK-NEXT:  %{{\d+}} = "riscv.add"(%{{\d+}}, %{{\d+}}) : (!riscv.ireg<t6>, !riscv.ireg<s0>) -> !riscv.ireg<t5>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 29 : i32} : () -> !riscv.ireg<t4>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 28 : i32} : () -> !riscv.ireg<t3>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 27 : i32} : () -> !riscv.ireg<s11>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 26 : i32} : () -> !riscv.ireg<s10>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 25 : i32} : () -> !riscv.ireg<s9>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 24 : i32} : () -> !riscv.ireg<s8>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 23 : i32} : () -> !riscv.ireg<s7>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 22 : i32} : () -> !riscv.ireg<s6>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 21 : i32} : () -> !riscv.ireg<s5>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 20 : i32} : () -> !riscv.ireg<s4>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 19 : i32} : () -> !riscv.ireg<s3>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 18 : i32} : () -> !riscv.ireg<s2>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 17 : i32} : () -> !riscv.ireg<a7>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 16 : i32} : () -> !riscv.ireg<a6>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 15 : i32} : () -> !riscv.ireg<a5>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 14 : i32} : () -> !riscv.ireg<a4>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 13 : i32} : () -> !riscv.ireg<a3>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 12 : i32} : () -> !riscv.ireg<a2>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 11 : i32} : () -> !riscv.ireg<a1>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 10 : i32} : () -> !riscv.ireg<a0>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 9 : i32} : () -> !riscv.ireg<s1>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 8 : i32} : () -> !riscv.ireg<t2>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 7 : i32} : () -> !riscv.ireg<t1>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.ireg<t0>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.ireg<ra>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 4 : i32} : () -> !riscv.ireg<j0>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 3 : i32} : () -> !riscv.ireg<j1>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 2 : i32} : () -> !riscv.ireg<j2>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.ireg<j3>
// CHECK-NEXT:  %{{\d+}} = "riscv.fcvt.s.w"(%{{\d+}}) : (!riscv.ireg<j2>) -> !riscv.freg<ft11>
// CHECK-NEXT:  %{{\d+}} = "riscv.fcvt.s.w"(%{{\d+}}) : (!riscv.ireg<j3>) -> !riscv.freg<ft10>
// CHECK-NEXT:  %{{\d+}} = "riscv.fadd.s"(%{{\d+}}, %{{\d+}}) : (!riscv.freg<ft11>, !riscv.freg<ft10>) -> !riscv.freg<ft9>
// CHECK-NEXT: }) : () -> ()
