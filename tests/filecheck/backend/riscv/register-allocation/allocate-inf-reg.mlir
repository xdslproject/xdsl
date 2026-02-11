// RUN: xdsl-opt --split-input-file -p "riscv-allocate-infinite-registers" %s | filecheck %s

builtin.module {
  riscv_func.func @main() {
    %0 = riscv.li 6 : !riscv.reg<j_0>
    %1 = riscv.li 5 : !riscv.reg<j_1>
    %2 = riscv.add %0, %1 : (!riscv.reg<j_0>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    %3 = riscv_scf.for %4 : !riscv.reg<j_0>  = %0 to %1 step %2 iter_args(%5 = %2) -> (!riscv.reg<j_2>) {
      riscv_scf.yield %5 : !riscv.reg<j_2>
    }
    riscv_func.return
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() {
// CHECK-NEXT:      %0 = riscv.li 6 : !riscv.reg<t0>
// CHECK-NEXT:      %1 = riscv.li 5 : !riscv.reg<s0>
// CHECK-NEXT:      %2 = riscv.add %0, %1 : (!riscv.reg<t0>, !riscv.reg<s0>) -> !riscv.reg<t1>
// CHECK-NEXT:      %3 = riscv_scf.for %4 : !riscv.reg<t0>  = %0 to %1 step %2 iter_args(%5 = %2) -> (!riscv.reg<t1>) {
// CHECK-NEXT:        %6 = riscv.mv %5 : (!riscv.reg<t1>) -> !riscv.reg<t1>
// CHECK-NEXT:        riscv_scf.yield %6 : !riscv.reg<t1>
// CHECK-NEXT:      }
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
