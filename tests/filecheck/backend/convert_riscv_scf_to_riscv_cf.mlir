// RUN: xdsl-opt -p convert-riscv-scf-to-riscv-cf --split-input-file %s | filecheck %s


builtin.module {
    riscv_func.func @copy10(%src : !riscv.reg<a0>, %dst : !riscv.reg<a1>) {
        %zero = riscv.li 0 : !riscv.reg<a2>
        %step = riscv.li 4 : !riscv.reg<a3>
        %forty = riscv.li 40 : !riscv.reg<a4>
        riscv_scf.for %offset : !riscv.reg<a5> = %zero to %forty step %step {
            %srcptr = riscv.add %src, %offset : (!riscv.reg<a0>, !riscv.reg<a5>) -> !riscv.reg<a6>
            %dstptr = riscv.add %dst, %offset : (!riscv.reg<a1>, !riscv.reg<a5>) -> !riscv.reg<a7>
            %val = riscv.lw %srcptr, 0 : (!riscv.reg<a6>) -> !riscv.reg<t0>
            riscv.sw %dstptr, %val, 0 : (!riscv.reg<a7>, !riscv.reg<t0>) -> ()
            yield
        }
        return
    }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @copy10(%src : !riscv.reg<a0>, %dst : !riscv.reg<a1>) {
// CHECK-NEXT:      %zero = riscv.li 0 : !riscv.reg<a2>
// CHECK-NEXT:      %step = riscv.li 4 : !riscv.reg<a3>
// CHECK-NEXT:      %forty = riscv.li 40 : !riscv.reg<a4>
// CHECK-NEXT:      %0 = riscv.mv %zero : (!riscv.reg<a2>) -> !riscv.reg<a5>
// CHECK-NEXT:      riscv_cf.bge %0 : !riscv.reg<a5>, %forty : !riscv.reg<a4>, ^0(%0 : !riscv.reg<a5>), ^1(%0 : !riscv.reg<a5>)
// CHECK-NEXT:    ^1(%offset : !riscv.reg<a5>):
// CHECK-NEXT:      riscv.label "scf_body_0_for"
// CHECK-NEXT:      %srcptr = riscv.add %src, %offset : (!riscv.reg<a0>, !riscv.reg<a5>) -> !riscv.reg<a6>
// CHECK-NEXT:      %dstptr = riscv.add %dst, %offset : (!riscv.reg<a1>, !riscv.reg<a5>) -> !riscv.reg<a7>
// CHECK-NEXT:      %val = riscv.lw %srcptr, 0 : (!riscv.reg<a6>) -> !riscv.reg<t0>
// CHECK-NEXT:      riscv.sw %dstptr, %val, 0 : (!riscv.reg<a7>, !riscv.reg<t0>) -> ()
// CHECK-NEXT:      %1 = riscv.add %offset, %step : (!riscv.reg<a5>, !riscv.reg<a3>) -> !riscv.reg<a5>
// CHECK-NEXT:      riscv_cf.blt %1 : !riscv.reg<a5>, %forty : !riscv.reg<a4>, ^1(%1 : !riscv.reg<a5>), ^0(%1 : !riscv.reg<a5>)
// CHECK-NEXT:    ^0(%2 : !riscv.reg<a5>):
// CHECK-NEXT:      riscv.label "scf_body_end_0_for"
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

// Python equivalent of below function in riscv_scf
// sum(range(arg0, arg1))

builtin.module {
    riscv_func.func @sum_range(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>) {
        %2 = riscv.li 1 : !riscv.reg<a2>
        %3 = riscv.li 0 : !riscv.reg<a3>
        %4 = riscv_scf.for %5 : !riscv.reg<a4> = %0 to %1 step %2 iter_args(%6 = %3) -> (!riscv.reg<a3>) {
            %7 = riscv.add %5, %6 : (!riscv.reg<a4>, !riscv.reg<a3>) -> !riscv.reg<a3>
            riscv_scf.yield %7 : !riscv.reg<a3>
        }
        %8 = riscv.mv %4 : (!riscv.reg<a3>) -> !riscv.reg<a0>
        riscv_func.return %8 : !riscv.reg<a0>
    }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @sum_range(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>) {
// CHECK-NEXT:      %2 = riscv.li 1 : !riscv.reg<a2>
// CHECK-NEXT:      %3 = riscv.li 0 : !riscv.reg<a3>
// CHECK-NEXT:      %4 = riscv.mv %0 : (!riscv.reg<a0>) -> !riscv.reg<a4>
// CHECK-NEXT:      riscv_cf.bge %4 : !riscv.reg<a4>, %1 : !riscv.reg<a1>, ^0(%4 : !riscv.reg<a4>, %3 : !riscv.reg<a3>), ^1(%4 : !riscv.reg<a4>, %3 : !riscv.reg<a3>)
// CHECK-NEXT:    ^1(%5 : !riscv.reg<a4>, %6 : !riscv.reg<a3>):
// CHECK-NEXT:      riscv.label "scf_body_0_for"
// CHECK-NEXT:      %7 = riscv.add %5, %6 : (!riscv.reg<a4>, !riscv.reg<a3>) -> !riscv.reg<a3>
// CHECK-NEXT:      %8 = riscv.add %5, %2 : (!riscv.reg<a4>, !riscv.reg<a2>) -> !riscv.reg<a4>
// CHECK-NEXT:      riscv_cf.blt %8 : !riscv.reg<a4>, %1 : !riscv.reg<a1>, ^1(%8 : !riscv.reg<a4>, %7 : !riscv.reg<a3>), ^0(%8 : !riscv.reg<a4>, %7 : !riscv.reg<a3>)
// CHECK-NEXT:    ^0(%9 : !riscv.reg<a4>, %10 : !riscv.reg<a3>):
// CHECK-NEXT:      riscv.label "scf_body_end_0_for"
// CHECK-NEXT:      %11 = riscv.mv %10 : (!riscv.reg<a3>) -> !riscv.reg<a0>
// CHECK-NEXT:      riscv_func.return %11 : !riscv.reg<a0>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

builtin.module {
    riscv_func.func @nested(%arg0 : !riscv.reg<a0>) {
        %0 = riscv.li 0 : !riscv.reg<a1>
        %1 = riscv.li 0 : !riscv.reg<a2>
        %2 = riscv.li 1 : !riscv.reg<a3>
        %3 = riscv_scf.for %arg1 : !riscv.reg<a2> = %1 to %arg0 step %2 iter_args(%arg2 = %0) -> (!riscv.reg<a1>) {
            %4 = riscv.li 0 : !riscv.reg<a4>
            %5 = riscv.li 1 : !riscv.reg<a5>
            %6 = riscv_scf.for %arg3 : !riscv.reg<a4> = %4 to %arg0 step %5 iter_args(%arg4 = %arg2) -> (!riscv.reg<a1>) {
                %7 = riscv.add %arg1, %arg3 : (!riscv.reg<a2>, !riscv.reg<a4>) -> !riscv.reg<a0>
                %8 = riscv.add %arg4, %7 : (!riscv.reg<a1>, !riscv.reg<a0>) -> !riscv.reg<a1>
                riscv_scf.yield %8 : !riscv.reg<a1>
            }
            riscv_scf.yield %6 : !riscv.reg<a1>
        }
        riscv_func.return %3 : !riscv.reg<a1>
    }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @nested(%arg0 : !riscv.reg<a0>) {
// CHECK-NEXT:      %0 = riscv.li 0 : !riscv.reg<a1>
// CHECK-NEXT:      %1 = riscv.li 0 : !riscv.reg<a2>
// CHECK-NEXT:      %2 = riscv.li 1 : !riscv.reg<a3>
// CHECK-NEXT:      %3 = riscv.mv %1 : (!riscv.reg<a2>) -> !riscv.reg<a2>
// CHECK-NEXT:      riscv_cf.bge %3 : !riscv.reg<a2>, %arg0 : !riscv.reg<a0>, ^0(%3 : !riscv.reg<a2>, %0 : !riscv.reg<a1>), ^1(%3 : !riscv.reg<a2>, %0 : !riscv.reg<a1>)
// CHECK-NEXT:    ^1(%arg1 : !riscv.reg<a2>, %arg2 : !riscv.reg<a1>):
// CHECK-NEXT:      riscv.label "scf_body_1_for"
// CHECK-NEXT:      %4 = riscv.li 0 : !riscv.reg<a4>
// CHECK-NEXT:      %5 = riscv.li 1 : !riscv.reg<a5>
// CHECK-NEXT:      %6 = riscv.mv %4 : (!riscv.reg<a4>) -> !riscv.reg<a4>
// CHECK-NEXT:      riscv_cf.bge %6 : !riscv.reg<a4>, %arg0 : !riscv.reg<a0>, ^2(%6 : !riscv.reg<a4>, %arg2 : !riscv.reg<a1>), ^3(%6 : !riscv.reg<a4>, %arg2 : !riscv.reg<a1>)
// CHECK-NEXT:    ^3(%arg3 : !riscv.reg<a4>, %arg4 : !riscv.reg<a1>):
// CHECK-NEXT:      riscv.label "scf_body_0_for"
// CHECK-NEXT:      %7 = riscv.add %arg1, %arg3 : (!riscv.reg<a2>, !riscv.reg<a4>) -> !riscv.reg<a0>
// CHECK-NEXT:      %8 = riscv.add %arg4, %7 : (!riscv.reg<a1>, !riscv.reg<a0>) -> !riscv.reg<a1>
// CHECK-NEXT:      %9 = riscv.add %arg3, %5 : (!riscv.reg<a4>, !riscv.reg<a5>) -> !riscv.reg<a4>
// CHECK-NEXT:      riscv_cf.blt %9 : !riscv.reg<a4>, %arg0 : !riscv.reg<a0>, ^3(%9 : !riscv.reg<a4>, %8 : !riscv.reg<a1>), ^2(%9 : !riscv.reg<a4>, %8 : !riscv.reg<a1>)
// CHECK-NEXT:    ^2(%10 : !riscv.reg<a4>, %11 : !riscv.reg<a1>):
// CHECK-NEXT:      riscv.label "scf_body_end_0_for"
// CHECK-NEXT:      %12 = riscv.add %arg1, %2 : (!riscv.reg<a2>, !riscv.reg<a3>) -> !riscv.reg<a2>
// CHECK-NEXT:      riscv_cf.blt %12 : !riscv.reg<a2>, %arg0 : !riscv.reg<a0>, ^1(%12 : !riscv.reg<a2>, %11 : !riscv.reg<a1>), ^0(%12 : !riscv.reg<a2>, %11 : !riscv.reg<a1>)
// CHECK-NEXT:    ^0(%13 : !riscv.reg<a2>, %14 : !riscv.reg<a1>):
// CHECK-NEXT:      riscv.label "scf_body_end_1_for"
// CHECK-NEXT:      riscv_func.return %14 : !riscv.reg<a1>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
