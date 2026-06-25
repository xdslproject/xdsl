// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive}" --disable-verify %s | filecheck %s

riscv_func.func @external() -> ()

riscv_func.func @main() {
  %0 = rv32.li 6 : !riscv.reg
  %1 = rv32.li 5 : !riscv.reg
  %5 = riscv.add %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
  riscv_cf.bge %0 : !riscv.reg, %1 : !riscv.reg, ^then(), ^else(%0 : !riscv.reg, %5 : !riscv.reg)

  ^else(%e0 : !riscv.reg, %e1 : !riscv.reg):
    riscv.label "else"
    %e2 = riscv.add %e0, %e1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    riscv_func.return

  ^then(%t0 : !riscv.reg, %t1 : !riscv.reg):
    riscv.label "then"
    %t2 = riscv.add %t0, %t1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    riscv_func.return
}

// CHECK:      builtin.module {
// CHECK-NEXT:   riscv_func.func @external() -> ()
// CHECK-NEXT:   riscv_func.func @main() {
// CHECK-NEXT:     %0 = rv32.li 6 : !riscv.reg<t0>
// CHECK-NEXT:     %1 = rv32.li 5 : !riscv.reg<t1>
// CHECK-NEXT:     %2 = riscv.add %0, %1 : (!riscv.reg<t0>, !riscv.reg<t1>) -> !riscv.reg<t2>
// CHECK-NEXT:     riscv_cf.bge %0 : !riscv.reg<t0>, %1 : !riscv.reg<t1>, ^then(), ^else(%0 : !riscv.reg<t0>, %2 : !riscv.reg<t2>)
// CHECK-NEXT:   ^else(%e0: !riscv.reg<t1>, %e1: !riscv.reg<t0>):
// CHECK-NEXT:     riscv.label "else"
// CHECK-NEXT:     %e2 = riscv.add %e0, %e1 : (!riscv.reg<t1>, !riscv.reg<t0>) -> !riscv.reg<t1>
// CHECK-NEXT:     riscv_func.return
// CHECK-NEXT:   ^then(%t0: !riscv.reg<t0>, %t1: !riscv.reg<t1>):
// CHECK-NEXT:     riscv.label "then"
// CHECK-NEXT:     %t2 = riscv.add %t0, %t1 : (!riscv.reg<t0>, !riscv.reg<t1>) -> !riscv.reg<t0>
// CHECK-NEXT:     riscv_func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
