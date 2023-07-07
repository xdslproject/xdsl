// RUN: xdsl-opt -p riscv-allocate-registers{allocation_strategy=LivenessBlockNaive limit_registers=2} %s --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<>
  %1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<s0>
  %2 = "riscv.add"(%0, %1) : (!riscv.reg<>, !riscv.reg<s0>) -> !riscv.reg<>
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
  %32 = "riscv.fcvt.s.w"(%30) : (!riscv.reg<>) -> !riscv.freg<>
  %33 = "riscv.fcvt.s.w"(%31) : (!riscv.reg<>) -> !riscv.freg<>
  %34 = "riscv.fadd.s"(%32, %33) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK:  %0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<j28>
// CHECK:  %1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<s0>
// CHECK:  %2 = "riscv.add"(%0, %1) : (!riscv.reg<j28>, !riscv.reg<s0>) -> !riscv.reg<j29>
// CHECK:  %3 = "riscv.li"() {"immediate" = 29 : i32} : () -> !riscv.reg<j27>
// CHECK:  %4 = "riscv.li"() {"immediate" = 28 : i32} : () -> !riscv.reg<j26>
// CHECK:  %5 = "riscv.li"() {"immediate" = 27 : i32} : () -> !riscv.reg<j25>
// CHECK:  %6 = "riscv.li"() {"immediate" = 26 : i32} : () -> !riscv.reg<j24>
// CHECK:  %7 = "riscv.li"() {"immediate" = 25 : i32} : () -> !riscv.reg<j23>
// CHECK:  %8 = "riscv.li"() {"immediate" = 24 : i32} : () -> !riscv.reg<j22>
// CHECK:  %9 = "riscv.li"() {"immediate" = 23 : i32} : () -> !riscv.reg<j21>
// CHECK:  %10 = "riscv.li"() {"immediate" = 22 : i32} : () -> !riscv.reg<j20>
// CHECK:  %11 = "riscv.li"() {"immediate" = 21 : i32} : () -> !riscv.reg<j19>
// CHECK:  %12 = "riscv.li"() {"immediate" = 20 : i32} : () -> !riscv.reg<j18>
// CHECK:  %13 = "riscv.li"() {"immediate" = 19 : i32} : () -> !riscv.reg<j17>
// CHECK:  %14 = "riscv.li"() {"immediate" = 18 : i32} : () -> !riscv.reg<j16>
// CHECK:  %15 = "riscv.li"() {"immediate" = 17 : i32} : () -> !riscv.reg<j15>
// CHECK:  %16 = "riscv.li"() {"immediate" = 16 : i32} : () -> !riscv.reg<j14>
// CHECK:  %17 = "riscv.li"() {"immediate" = 15 : i32} : () -> !riscv.reg<j13>
// CHECK:  %18 = "riscv.li"() {"immediate" = 14 : i32} : () -> !riscv.reg<j12>
// CHECK:  %19 = "riscv.li"() {"immediate" = 13 : i32} : () -> !riscv.reg<j11>
// CHECK:  %20 = "riscv.li"() {"immediate" = 12 : i32} : () -> !riscv.reg<j10>
// CHECK:  %21 = "riscv.li"() {"immediate" = 11 : i32} : () -> !riscv.reg<j9>
// CHECK:  %22 = "riscv.li"() {"immediate" = 10 : i32} : () -> !riscv.reg<j8>
// CHECK:  %23 = "riscv.li"() {"immediate" = 9 : i32} : () -> !riscv.reg<j7>
// CHECK:  %24 = "riscv.li"() {"immediate" = 8 : i32} : () -> !riscv.reg<j6>
// CHECK:  %25 = "riscv.li"() {"immediate" = 7 : i32} : () -> !riscv.reg<j5>
// CHECK:  %26 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<j4>
// CHECK:  %27 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<j3>
// CHECK:  %28 = "riscv.li"() {"immediate" = 4 : i32} : () -> !riscv.reg<j2>
// CHECK:  %29 = "riscv.li"() {"immediate" = 3 : i32} : () -> !riscv.reg<j1>
// CHECK:  %30 = "riscv.li"() {"immediate" = 2 : i32} : () -> !riscv.reg<ra>
// CHECK:  %31 = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<t0>
// CHECK:  %32 = "riscv.fcvt.s.w"(%30) : (!riscv.reg<ra>) -> !riscv.freg<ft1>
// CHECK:  %33 = "riscv.fcvt.s.w"(%31) : (!riscv.reg<t0>) -> !riscv.freg<ft0>
// CHECK:  %34 = "riscv.fadd.s"(%32, %33) : (!riscv.freg<ft1>, !riscv.freg<ft0>) -> !riscv.freg<j0>
// CHECK:}) : () -> ()
