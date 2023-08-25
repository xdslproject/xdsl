// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive limit_registers=2}" %s --print-op-generic | filecheck %s

builtin.module {
  riscv_func.func @main() {
    %0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<>
    %1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<s0>
    %2 = "riscv.add"(%0, %1) : (!riscv.reg<>, !riscv.reg<s0>) -> !riscv.reg<>
    %3 = "riscv.li"() {"immediate" = 29 : i32} : () -> !riscv.reg<>
    %4 = "riscv.li"() {"immediate" = 28 : i32} : () -> !riscv.reg<>
    %5 = "riscv.add"(%3, %4) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %6 = "riscv.li"() {"immediate" = 26 : i32} : () -> !riscv.reg<>
    %7 = "riscv.li"() {"immediate" = 25 : i32} : () -> !riscv.reg<>
    %8 = "riscv.li"() {"immediate" = 24 : i32} : () -> !riscv.reg<>
    %9 = "riscv.li"() {"immediate" = 23 : i32} : () -> !riscv.reg<>
    %10 = "riscv.li"() {"immediate" = 2 : i32} : () -> !riscv.reg<>
    %11 = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<>
    %12 = "riscv.fcvt.s.w"(%10) : (!riscv.reg<>) -> !riscv.freg<>
    %13 = "riscv.fcvt.s.w"(%11) : (!riscv.reg<>) -> !riscv.freg<>
    %14 = "riscv.fadd.s"(%12, %13) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    riscv_func.return
  }
}
// CHECK:     "builtin.module"() ({
// CHECK-NEXT:  "riscv_func.func"() ({
// CHECK-NEXT:     %0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<ra>
// CHECK-NEXT:     %1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<s0>
// CHECK-NEXT:     %2 = "riscv.add"(%0, %1) : (!riscv.reg<ra>, !riscv.reg<s0>) -> !riscv.reg<t0>
// CHECK-NEXT:     %3 = "riscv.li"() {"immediate" = 29 : i32} : () -> !riscv.reg<ra>
// CHECK-NEXT:     %4 = "riscv.li"() {"immediate" = 28 : i32} : () -> !riscv.reg<t0>
// CHECK-NEXT:     %5 = "riscv.add"(%3, %4) : (!riscv.reg<ra>, !riscv.reg<t0>) -> !riscv.reg<j1>
// CHECK-NEXT:     %6 = "riscv.li"() {"immediate" = 26 : i32} : () -> !riscv.reg<ra>
// CHECK-NEXT:     %7 = "riscv.li"() {"immediate" = 25 : i32} : () -> !riscv.reg<ra>
// CHECK-NEXT:     %8 = "riscv.li"() {"immediate" = 24 : i32} : () -> !riscv.reg<ra>
// CHECK-NEXT:     %9 = "riscv.li"() {"immediate" = 23 : i32} : () -> !riscv.reg<ra>
// CHECK-NEXT:     %10 = "riscv.li"() {"immediate" = 2 : i32} : () -> !riscv.reg<ra>
// CHECK-NEXT:     %11 = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<t0>
// CHECK-NEXT:     %12 = "riscv.fcvt.s.w"(%10) : (!riscv.reg<ra>) -> !riscv.freg<ft1>
// CHECK-NEXT:     %13 = "riscv.fcvt.s.w"(%11) : (!riscv.reg<t0>) -> !riscv.freg<ft0>
// CHECK-NEXT:     %14 = "riscv.fadd.s"(%12, %13) : (!riscv.freg<ft1>, !riscv.freg<ft0>) -> !riscv.freg<j0>
// CHECK-NEXT:    "riscv_func.return"() : () -> ()
// CHECK-NEXT:  }) {"sym_name" = "main", "function_type" = () -> ()} : () -> ()
// CHECK:     }) : () -> ()
