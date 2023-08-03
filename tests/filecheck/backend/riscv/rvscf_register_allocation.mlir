"riscv_func.func"() ({
^0(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>):
  %2 = riscv.mv %0 : (!riscv.reg<a0>) -> !riscv.reg<>
  %3 = riscv.mv %1 : (!riscv.reg<a1>) -> !riscv.reg<>
  %4 = riscv.li {"immediate" = 1 : si32} : () -> !riscv.reg<>
  %5 = riscv.li {"immediate" = 0 : si32} : () -> !riscv.reg<>
  %6 = "riscv_scf.for"(%2, %3, %4, %5) ({
  ^1(%7 : !riscv.reg<>, %8 : !riscv.reg<>):
    %9 = riscv.add %8, %7 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    "riscv_scf.yield"(%9) : (!riscv.reg<>) -> ()
  }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %10 = riscv.mv %6 : (!riscv.reg<>) -> !riscv.reg<a0>
  "riscv_func.return"(%10) : (!riscv.reg<a0>) -> ()
}) {"sym_name" = "foo"} : () -> ()

// CHECK:       "riscv_func.func"() ({
// CHECK-NEXT:  ^0(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>):
// CHECK-NEXT:    %2 = riscv.mv %0 : (!riscv.reg<a0>) -> !riscv.reg<a0>
// CHECK-NEXT:    %3 = riscv.mv %1 : (!riscv.reg<a1>) -> !riscv.reg<a1>
// CHECK-NEXT:    %4 = riscv.li {"immediate" = 1 : si32} : () -> !riscv.reg<a2>
// CHECK-NEXT:    %5 = riscv.li {"immediate" = 0 : si32} : () -> !riscv.reg<a3>
// CHECK-NEXT:    %6 = "riscv_scf.for"(%2, %3, %4, %5) ({
// CHECK-NEXT:    ^1(%7 : !riscv.reg<a0>, %8 : !riscv.reg<a3>):
// CHECK-NEXT:      %9 = riscv.add %8, %7 : (!riscv.reg<a3>, !riscv.reg<a0>) -> !riscv.reg<a3>
// CHECK-NEXT:      "riscv_scf.yield"(%9) : (!riscv.reg<a3>) -> ()
// CHECK-NEXT:    }) : (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>) -> !riscv.reg<a3>
// CHECK-NEXT:    %10 = riscv.mv %6 : (!riscv.reg<a3>) -> !riscv.reg<a0>
// CHECK-NEXT:    "riscv_func.return"(%10) : (!riscv.reg<a0>) -> ()
// CHECK-NEXT:  }) {"sym_name" = "foo"} : () -> ()