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
// CHECK-NEXT:   %4 = builtin.unrealized_conversion_cast %0 : index to !riscv.reg<>
// CHECK-NEXT:   %5 = builtin.unrealized_conversion_cast %1 : index to !riscv.reg<>
// CHECK-NEXT:   %6 = builtin.unrealized_conversion_cast %2 : index to !riscv.reg<>
// CHECK-NEXT:   %7 = builtin.unrealized_conversion_cast %3 : index to !riscv.reg<>
// CHECK-NEXT:   %8 = riscv.mv %7 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %9 = "riscv_scf.for"(%4, %5, %6, %8) ({
// CHECK-NEXT:   ^0(%10 : !riscv.reg<>, %11 : !riscv.reg<>):
// CHECK-NEXT:     %12 = builtin.unrealized_conversion_cast %11 : !riscv.reg<> to index
// CHECK-NEXT:     %13 = builtin.unrealized_conversion_cast %10 : !riscv.reg<> to index
// CHECK-NEXT:     %14 = arith.addi %13, %12 : index
// CHECK-NEXT:     %15 = builtin.unrealized_conversion_cast %14 : index to !riscv.reg<>
// CHECK-NEXT:     "riscv_scf.yield"(%15) : (!riscv.reg<>) -> ()
// CHECK-NEXT:   }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %16 = builtin.unrealized_conversion_cast %9 : !riscv.reg<> to index
// CHECK-NEXT: }
