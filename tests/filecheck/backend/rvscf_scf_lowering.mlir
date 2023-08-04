// RUN: xdsl-opt -p scf-to-rvscf-lowering %s | filecheck %s

builtin.module {
  %0 = arith.constant 0 : index
  %1 = arith.constant 10 : index
  %2 = arith.constant 1 : index
  %3 = arith.constant 0 : index
  %4 = "scf.for"(%0, %1, %2, %3) ({
  ^0(%5 : index, %6 : index):
    %7 = arith.addi %5, %6 : index
    "scf.yield"(%7) : (index) -> ()
  }) : (index, index, index, index) -> index
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 10 : index
// CHECK-NEXT:   %2 = arith.constant 1 : index
// CHECK-NEXT:   %3 = arith.constant 0 : index
// CHECK-NEXT:   %4 = "builtin.unrealized_conversion_cast"(%0) : (index) -> !riscv.reg<>
// CHECK-NEXT:   %5 = "builtin.unrealized_conversion_cast"(%1) : (index) -> !riscv.reg<>
// CHECK-NEXT:   %6 = "builtin.unrealized_conversion_cast"(%2) : (index) -> !riscv.reg<>
// CHECK-NEXT:   %7 = "builtin.unrealized_conversion_cast"(%3) : (index) -> !riscv.reg<>
// CHECK-NEXT:   %8 = "riscv_scf.for"(%4, %5, %6, %7) ({
// CHECK-NEXT:   ^0(%9 : !riscv.reg<>, %10 : !riscv.reg<>):
// CHECK-NEXT:     %11 = "builtin.unrealized_conversion_cast"(%10) : (!riscv.reg<>) -> index
// CHECK-NEXT:     %12 = "builtin.unrealized_conversion_cast"(%9) : (!riscv.reg<>) -> index
// CHECK-NEXT:     %13 = arith.addi %12, %11 : index
// CHECK-NEXT:     %14 = "builtin.unrealized_conversion_cast"(%13) : (index) -> !riscv.reg<>
// CHECK-NEXT:     "riscv_scf.yield"(%14) : (!riscv.reg<>) -> ()
// CHECK-NEXT:   }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %15 = "builtin.unrealized_conversion_cast"(%8) : (!riscv.reg<>) -> index
// CHECK-NEXT: }

