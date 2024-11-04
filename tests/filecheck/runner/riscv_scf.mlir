// RUN: xdsl-run --verbose %s | filecheck %s
// RUN: xdsl-run --verbose --symbol="sum_to" --args="6" %s | filecheck %s --check-prefix=CHECK-ARGS

builtin.module {
  func.func @sum_to(%0 : !riscv.reg) -> !riscv.reg {
    %1 = riscv.li 0 : !riscv.reg
    %2 = riscv.li 1 : !riscv.reg
    %3 = riscv.li 0 : !riscv.reg
    %4 = riscv_scf.for %5 : !riscv.reg = %1 to %0 step %2 iter_args(%6 = %3) -> (!riscv.reg) {
      %7 = riscv.add %5, %6 : (!riscv.reg, !riscv.reg) -> !riscv.reg
      riscv_scf.yield %7 : !riscv.reg
    }
    func.return %4 : !riscv.reg
  }
  func.func @main() -> !riscv.reg {
    %c5 = riscv.li 5 : !riscv.reg
    %res = func.call @sum_to(%c5) : (!riscv.reg) -> !riscv.reg
    return %res : !riscv.reg
  }
}

// CHECK:       result: 10
// CHECK-ARGS:  result: 15
