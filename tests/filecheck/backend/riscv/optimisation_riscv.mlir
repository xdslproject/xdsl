// RUN: xdsl-opt -p optimise-riscv %s | filecheck %s

builtin.module {
  "riscv_func.func"() ({
  ^0(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>):
    %2 = riscv.mv %0 : (!riscv.reg<a0>) -> !riscv.reg<a0>
    %3 = riscv.mv %1 : (!riscv.reg<a1>) -> !riscv.reg<a1>
    %4 = riscv.li 1 : () -> !riscv.reg<t0>
    %5 = riscv.li 1 : () -> !riscv.reg<t1>
    %6 = "riscv_scf.for"(%2, %3, %4, %5) ({
    ^1(%7 : !riscv.reg<a0>, %8 : !riscv.reg<t1>):
      %9 = riscv.add %8, %7 : (!riscv.reg<t1>, !riscv.reg<a0>) -> !riscv.reg<t1>
      "riscv_scf.yield"(%9) : (!riscv.reg<t1>) -> ()
    }) : (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<t0>, !riscv.reg<t1>) -> !riscv.reg<t1>
    %10 = riscv.mv %6 : (!riscv.reg<t1>) -> !riscv.reg<a0>
    "riscv_func.return"(%10) : (!riscv.reg<a0>) -> ()
  }) {"sym_name" = "funky"} : () -> ()
}

// CHECK: builtin.module {
// CHECK-NEXT:   "riscv_func.func"() ({
// CHECK-NEXT:   ^0(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>):
// CHECK-NEXT:     %2 = riscv.li 1 : () -> !riscv.reg<t0>
// CHECK-NEXT:     %3 = riscv.li 1 : () -> !riscv.reg<t1>
// CHECK-NEXT:     %4 = "riscv_scf.for"(%0, %1, %2, %3) ({
// CHECK-NEXT:     ^1(%5 : !riscv.reg<a0>, %6 : !riscv.reg<t1>):
// CHECK-NEXT:       %7 = riscv.add %6, %5 : (!riscv.reg<t1>, !riscv.reg<a0>) -> !riscv.reg<t1>
// CHECK-NEXT:       "riscv_scf.yield"(%7) : (!riscv.reg<t1>) -> ()
// CHECK-NEXT:     }) : (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<t0>, !riscv.reg<t1>) -> !riscv.reg<t1>
// CHECK-NEXT:     %8 = riscv.mv %4 : (!riscv.reg<t1>) -> !riscv.reg<a0>
// CHECK-NEXT:      "riscv_func.return"(%8) : (!riscv.reg<a0>) -> ()
// CHECK-NEXT:   }) {"sym_name" = "funky"} : () -> ()
// CHECK-NEXT: }
