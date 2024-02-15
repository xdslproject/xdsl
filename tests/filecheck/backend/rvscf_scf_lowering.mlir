// RUN: xdsl-opt -p convert-scf-to-riscv-scf %s | filecheck %s

builtin.module {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %i_in = arith.constant 42 : index
  %f_in = arith.constant 42.0 : f64

  %i_out, %f_out = scf.for %idx = %c0 to %c10 step %c1 iter_args(%i_acc = %i_in, %f_acc = %f_in) -> (index, f64) {
    %res = arith.addi %idx, %i_acc : index
    scf.yield %res, %f_acc : index, f64
  }
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
// CHECK-NEXT:   %9 = riscv_scf.for %10 : !riscv.reg<> = %4 to %5 step %6 iter_args(%11 = %8) -> (!riscv.reg<>) {
// CHECK-NEXT:     %12 = builtin.unrealized_conversion_cast %11 : !riscv.reg<> to index
// CHECK-NEXT:     %13 = builtin.unrealized_conversion_cast %10 : !riscv.reg<> to index
// CHECK-NEXT:     %14 = arith.addi %13, %12 : index
// CHECK-NEXT:     %15 = builtin.unrealized_conversion_cast %14 : index to !riscv.reg<>
// CHECK-NEXT:     riscv_scf.yield %15 : !riscv.reg<>
// CHECK-NEXT:   }
// CHECK-NEXT:   %16 = builtin.unrealized_conversion_cast %9 : !riscv.reg<> to index
// CHECK-NEXT: }
