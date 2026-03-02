// RUN: xdsl-opt -p riscv-allocate-registers,convert-riscv-scf-to-riscv-cf %s | filecheck %s

builtin.module {
    riscv_func.func @sum_range(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>) {
        %2 = rv32.li 1 : !riscv.reg
        %3 = rv32.li 0 : !riscv.reg
        %arg = riscv.mv %3 : (!riscv.reg) -> !riscv.reg
        %4 = riscv_scf.for %5 : !riscv.reg = %0 to %1 step %2 iter_args(%6 = %arg) -> (!riscv.reg) {
            %7 = riscv.add %5, %6 : (!riscv.reg, !riscv.reg) -> !riscv.reg
            riscv_scf.yield %7 : !riscv.reg
        }
        %8 = riscv.mv %4 : (!riscv.reg) -> !riscv.reg
        riscv_func.return %8 : !riscv.reg
    }
}

// CHECK:        builtin.module {
// CHECK-NEXT:    riscv_func.func @sum_range(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>) {
// CHECK-NEXT:      %2 = rv32.li 1 : !riscv.reg<t2>
// CHECK-NEXT:      %3 = rv32.li 0 : !riscv.reg<zero>
// CHECK-NEXT:      %arg = riscv.mv %3 : (!riscv.reg<zero>) -> !riscv.reg<t0>
// CHECK-NEXT:      %4 = riscv.mv %0 : (!riscv.reg<a0>) -> !riscv.reg<t1>
// CHECK-NEXT:      riscv_cf.bge %4 : !riscv.reg<t1>, %1 : !riscv.reg<a1>, ^bb0(%4 : !riscv.reg<t1>, %arg : !riscv.reg<t0>), ^bb1(%4 : !riscv.reg<t1>, %arg : !riscv.reg<t0>)
// CHECK-NEXT:    ^bb1(%5 : !riscv.reg<t1>, %6 : !riscv.reg<t0>):
// CHECK-NEXT:      riscv.label "scf_body_0_for"
// CHECK-NEXT:      %7 = riscv.add %5, %6 : (!riscv.reg<t1>, !riscv.reg<t0>) -> !riscv.reg<t0>
// CHECK-NEXT:      %8 = riscv.add %5, %2 : (!riscv.reg<t1>, !riscv.reg<t2>) -> !riscv.reg<t1>
// CHECK-NEXT:      riscv_cf.blt %8 : !riscv.reg<t1>, %1 : !riscv.reg<a1>, ^bb1(%8 : !riscv.reg<t1>, %7 : !riscv.reg<t0>), ^bb0(%8 : !riscv.reg<t1>, %7 : !riscv.reg<t0>)
// CHECK-NEXT:    ^bb0(%9 : !riscv.reg<t1>, %10 : !riscv.reg<t0>):
// CHECK-NEXT:      riscv.label "scf_body_end_0_for"
// CHECK-NEXT:      %11 = riscv.mv %10 : (!riscv.reg<t0>) -> !riscv.reg<t0>
// CHECK-NEXT:      riscv_func.return %11 : !riscv.reg<t0>
// CHECK-NEXT:    }
// CHECK-NEXT:   }
