// RUN: xdsl-opt -p riscv-allocate-registers{allocation_type=BlockNaive} %s --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<>
  %1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<s0>
  %2 = "riscv.add"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %3 = "riscv.li"() {"immediate" = 29 : i32} : () -> !riscv.reg<>
  %4 = "riscv.li"() {"immediate" = 28 : i32} : () -> !riscv.reg<>
  %5 = "riscv.li"() {"immediate" = 27 : i32} : () -> !riscv.reg<>
  %6 = "riscv.li"() {"immediate" = 26 : i32} : () -> !riscv.reg<>
  %7 = "riscv.li"() {"immediate" = 25 : i32} : () -> !riscv.reg<>
  %8 = "riscv.li"() {"immediate" = 24 : i32} : () -> !riscv.reg<>
  %9 = "riscv.li"() {"immediate" = 23 : i32} : () -> !riscv.reg<>
  %10 = "riscv.li"() {"immediate" = 22 : i32} : () -> !riscv.reg<>
  %11 = "riscv.li"() {"immediate" = 21 : i32} : () -> !riscv.reg<>
  %12 = "riscv.li"() {"immediate" = 20 : i32} : () -> !riscv.reg<>
  %13 = "riscv.li"() {"immediate" = 19 : i32} : () -> !riscv.reg<>
  %14 = "riscv.li"() {"immediate" = 18 : i32} : () -> !riscv.reg<>
  %15 = "riscv.li"() {"immediate" = 17 : i32} : () -> !riscv.reg<>
  %16 = "riscv.li"() {"immediate" = 16 : i32} : () -> !riscv.reg<>
  %17 = "riscv.li"() {"immediate" = 15 : i32} : () -> !riscv.reg<>
  %18 = "riscv.li"() {"immediate" = 14 : i32} : () -> !riscv.reg<>
  %19 = "riscv.li"() {"immediate" = 13 : i32} : () -> !riscv.reg<>
  %20 = "riscv.li"() {"immediate" = 12 : i32} : () -> !riscv.reg<>
  %21 = "riscv.li"() {"immediate" = 11 : i32} : () -> !riscv.reg<>
  %22 = "riscv.li"() {"immediate" = 10 : i32} : () -> !riscv.reg<>
  %23 = "riscv.li"() {"immediate" = 9 : i32} : () -> !riscv.reg<>
  %24 = "riscv.li"() {"immediate" = 8 : i32} : () -> !riscv.reg<>
  %25 = "riscv.li"() {"immediate" = 7 : i32} : () -> !riscv.reg<>
  %26 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<>
  %27 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<>
  %28 = "riscv.li"() {"immediate" = 4 : i32} : () -> !riscv.reg<>
  %29 = "riscv.li"() {"immediate" = 3 : i32} : () -> !riscv.reg<>
  %30 = "riscv.li"() {"immediate" = 2 : i32} : () -> !riscv.reg<>
  %31 = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<>
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<t0>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<s0>
// CHECK-NEXT:  %{{\d+}} = "riscv.add"(%{{\d+}}, %{{\d+}}) : (!riscv.reg<t0>, !riscv.reg<s0>) -> !riscv.reg<t1>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 29 : i32} : () -> !riscv.reg<t2>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 28 : i32} : () -> !riscv.reg<s1>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 27 : i32} : () -> !riscv.reg<a0>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 26 : i32} : () -> !riscv.reg<a1>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 25 : i32} : () -> !riscv.reg<a2>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 24 : i32} : () -> !riscv.reg<a3>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 23 : i32} : () -> !riscv.reg<a4>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 22 : i32} : () -> !riscv.reg<a5>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 21 : i32} : () -> !riscv.reg<a6>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 20 : i32} : () -> !riscv.reg<a7>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 19 : i32} : () -> !riscv.reg<s2>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 18 : i32} : () -> !riscv.reg<s3>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 17 : i32} : () -> !riscv.reg<s4>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 16 : i32} : () -> !riscv.reg<s5>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 15 : i32} : () -> !riscv.reg<s6>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 14 : i32} : () -> !riscv.reg<s7>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 13 : i32} : () -> !riscv.reg<s8>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 12 : i32} : () -> !riscv.reg<s9>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 11 : i32} : () -> !riscv.reg<s10>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 10 : i32} : () -> !riscv.reg<s11>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 9 : i32} : () -> !riscv.reg<t3>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 8 : i32} : () -> !riscv.reg<t4>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 7 : i32} : () -> !riscv.reg<t5>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<t6>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<j0>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 4 : i32} : () -> !riscv.reg<j1>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 3 : i32} : () -> !riscv.reg<j2>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 2 : i32} : () -> !riscv.reg<j3>
// CHECK-NEXT:  %{{\d+}} = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<j4>
// CHECK-NEXT: }) : () -> ()
