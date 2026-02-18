// RUN: xdsl-opt -p lower-riscv-scf-to-labels --split-input-file %s | filecheck %s

// sum(range(arg0, arg1))

builtin.module {
    riscv_func.func @foo(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>) {
        %2 = rv32.li 1 : !riscv.reg<a2>
        %3 = rv32.li 0 : !riscv.reg<a3>
        %4 = riscv_scf.for %5 : !riscv.reg<a4> = %0 to %1 step %2 iter_args(%6 = %3) -> (!riscv.reg<a3>) {
            %7 = riscv.add %5, %6 : (!riscv.reg<a4>, !riscv.reg<a3>) -> !riscv.reg<a3>
            "riscv_scf.yield"(%7) : (!riscv.reg<a3>) -> ()
        }
        %8 = riscv.mv %4 : (!riscv.reg<a3>) -> !riscv.reg<a0>
        riscv_func.return %8 : !riscv.reg<a0>
    }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @foo(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>) {
// CHECK-NEXT:      %2 = rv32.li 1 : !riscv.reg<a2>
// CHECK-NEXT:      %3 = rv32.li 0 : !riscv.reg<a3>
// CHECK-NEXT:      %4 = riscv.mv %0 : (!riscv.reg<a0>) -> !riscv.reg<a4>
// CHECK-NEXT:      riscv.label "scf_cond_0_for"
// CHECK-NEXT:      riscv.bge %4, %1, "scf_body_end_0_for" : (!riscv.reg<a4>, !riscv.reg<a1>) -> ()
// CHECK-NEXT:      riscv.label "scf_body_0_for"
// CHECK-NEXT:      %5 = rv32.get_register : !riscv.reg<a3>
// CHECK-NEXT:      %6 = rv32.get_register : !riscv.reg<a4>
// CHECK-NEXT:      %7 = riscv.add %6, %5 : (!riscv.reg<a4>, !riscv.reg<a3>) -> !riscv.reg<a3>
// CHECK-NEXT:      %8 = riscv.add %4, %2 : (!riscv.reg<a4>, !riscv.reg<a2>) -> !riscv.reg<a4>
// CHECK-NEXT:      riscv.blt %4, %1, "scf_body_0_for" : (!riscv.reg<a4>, !riscv.reg<a1>) -> ()
// CHECK-NEXT:      riscv.label "scf_body_end_0_for"
// CHECK-NEXT:      %9 = rv32.get_register : !riscv.reg<a3>
// CHECK-NEXT:      %10 = riscv.mv %9 : (!riscv.reg<a3>) -> !riscv.reg<a0>
// CHECK-NEXT:      riscv_func.return %10 : !riscv.reg<a0>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// -----

// sum(range(arg0, arg1))

  builtin.module {
    riscv_func.func @foo(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>) {
        %2 = rv32.li 1 : !riscv.reg<a2>
        %3 = rv32.li 0 : !riscv.reg<a3>
        %4 = riscv.fcvt.s.w %3 : (!riscv.reg<a3>) -> !riscv.freg<fa0>
        %5 = riscv_scf.for %6 : !riscv.reg<a0> = %0 to %1 step %2 iter_args(%7 = %4) -> (!riscv.freg<fa0>) {
            %8 = riscv.fcvt.s.w %6 : (!riscv.reg<a0>) -> !riscv.freg<fa1>
            %9 = riscv.fadd.s %7, %8 : (!riscv.freg<fa0>, !riscv.freg<fa1>) -> !riscv.freg<fa0>
            "riscv_scf.yield"(%9) : (!riscv.freg<fa0>) -> ()
        }
        %10 = riscv.fcvt.w.s %5 : (!riscv.freg<fa0>) -> !riscv.reg<a0>
        riscv_func.return %10 : !riscv.reg<a0>
    }
  }

// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @foo(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>) {
// CHECK-NEXT:      %2 = rv32.li 1 : !riscv.reg<a2>
// CHECK-NEXT:      %3 = rv32.li 0 : !riscv.reg<a3>
// CHECK-NEXT:      %4 = riscv.fcvt.s.w %3 : (!riscv.reg<a3>) -> !riscv.freg<fa0>
// CHECK-NEXT:      %5 = riscv.mv %0 : (!riscv.reg<a0>) -> !riscv.reg<a0>
// CHECK-NEXT:      riscv.label "scf_cond_0_for"
// CHECK-NEXT:      riscv.bge %5, %1, "scf_body_end_0_for" : (!riscv.reg<a0>, !riscv.reg<a1>) -> ()
// CHECK-NEXT:      riscv.label "scf_body_0_for"
// CHECK-NEXT:      %6 = riscv.get_float_register : !riscv.freg<fa0>
// CHECK-NEXT:      %7 = rv32.get_register : !riscv.reg<a0>
// CHECK-NEXT:      %8 = riscv.fcvt.s.w %7 : (!riscv.reg<a0>) -> !riscv.freg<fa1>
// CHECK-NEXT:      %9 = riscv.fadd.s %6, %8 : (!riscv.freg<fa0>, !riscv.freg<fa1>) -> !riscv.freg<fa0>
// CHECK-NEXT:      %10 = riscv.add %5, %2 : (!riscv.reg<a0>, !riscv.reg<a2>) -> !riscv.reg<a0>
// CHECK-NEXT:      riscv.blt %5, %1, "scf_body_0_for" : (!riscv.reg<a0>, !riscv.reg<a1>) -> ()
// CHECK-NEXT:      riscv.label "scf_body_end_0_for"
// CHECK-NEXT:      %11 = riscv.get_float_register : !riscv.freg<fa0>
// CHECK-NEXT:      %12 = riscv.fcvt.w.s %11 : (!riscv.freg<fa0>) -> !riscv.reg<a0>
// CHECK-NEXT:      riscv_func.return %12 : !riscv.reg<a0>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

builtin.module {
    riscv_func.func @foo(%arg0 : !riscv.reg<a0>) {
        %0 = rv32.li 0 : !riscv.reg<a1>
        %1 = rv32.li 0 : !riscv.reg<a2>
        %2 = rv32.li 1 : !riscv.reg<a3>
        %3 = riscv_scf.for %arg1 : !riscv.reg<a2> = %1 to %arg0 step %2 iter_args(%arg2 = %0) -> (!riscv.reg<a1>) {
            %4 = rv32.li 0 : !riscv.reg<a4>
            %5 = rv32.li 1 : !riscv.reg<a5>
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
// CHECK-NEXT:    riscv_func.func @foo(%arg0 : !riscv.reg<a0>) {
// CHECK-NEXT:      %0 = rv32.li 0 : !riscv.reg<a1>
// CHECK-NEXT:      %1 = rv32.li 0 : !riscv.reg<a2>
// CHECK-NEXT:      %2 = rv32.li 1 : !riscv.reg<a3>
// CHECK-NEXT:      %3 = riscv.mv %1 : (!riscv.reg<a2>) -> !riscv.reg<a2>
// CHECK-NEXT:      riscv.label "scf_cond_0_for"
// CHECK-NEXT:      riscv.bge %3, %arg0, "scf_body_end_0_for" : (!riscv.reg<a2>, !riscv.reg<a0>) -> ()
// CHECK-NEXT:      riscv.label "scf_body_0_for"
// CHECK-NEXT:      %arg2 = rv32.get_register : !riscv.reg<a1>
// CHECK-NEXT:      %arg1 = rv32.get_register : !riscv.reg<a2>
// CHECK-NEXT:      %4 = rv32.li 0 : !riscv.reg<a4>
// CHECK-NEXT:      %5 = rv32.li 1 : !riscv.reg<a5>
// CHECK-NEXT:      %6 = riscv.mv %4 : (!riscv.reg<a4>) -> !riscv.reg<a4>
// CHECK-NEXT:      riscv.label "scf_cond_1_for"
// CHECK-NEXT:      riscv.bge %6, %arg0, "scf_body_end_1_for" : (!riscv.reg<a4>, !riscv.reg<a0>) -> ()
// CHECK-NEXT:      riscv.label "scf_body_1_for"
// CHECK-NEXT:      %arg4 = rv32.get_register : !riscv.reg<a1>
// CHECK-NEXT:      %arg3 = rv32.get_register : !riscv.reg<a4>
// CHECK-NEXT:      %7 = riscv.add %arg1, %arg3 : (!riscv.reg<a2>, !riscv.reg<a4>) -> !riscv.reg<a0>
// CHECK-NEXT:      %8 = riscv.add %arg4, %7 : (!riscv.reg<a1>, !riscv.reg<a0>) -> !riscv.reg<a1>
// CHECK-NEXT:      %9 = riscv.add %6, %5 : (!riscv.reg<a4>, !riscv.reg<a5>) -> !riscv.reg<a4>
// CHECK-NEXT:      riscv.blt %6, %arg0, "scf_body_1_for" : (!riscv.reg<a4>, !riscv.reg<a0>) -> ()
// CHECK-NEXT:      riscv.label "scf_body_end_1_for"
// CHECK-NEXT:      %10 = rv32.get_register : !riscv.reg<a1>
// CHECK-NEXT:      %11 = riscv.add %3, %2 : (!riscv.reg<a2>, !riscv.reg<a3>) -> !riscv.reg<a2>
// CHECK-NEXT:      riscv.blt %3, %arg0, "scf_body_0_for" : (!riscv.reg<a2>, !riscv.reg<a0>) -> ()
// CHECK-NEXT:      riscv.label "scf_body_end_0_for"
// CHECK-NEXT:      %12 = rv32.get_register : !riscv.reg<a1>
// CHECK-NEXT:      riscv_func.return %12 : !riscv.reg<a1>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
