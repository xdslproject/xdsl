// RUN: XDSL_ROUNDTRIP

%lb = "rv32.li"() {"immediate" = 0: i32} : () -> !riscv.reg
%ub = "rv32.li"() {"immediate" = 100: i32} : () -> !riscv.reg
%step = "rv32.li"() {"immediate" = 1: i32} : () -> !riscv.reg
%acc = "rv32.li"() {"immediate" = 0 : i32} : () -> !riscv.reg<t0>
riscv_scf.for %i : !riscv.reg = %lb to %ub step %step {
    riscv.addi %acc, 1 : (!riscv.reg<t0>) -> !riscv.reg<t0>
}
riscv_scf.rof %j : !riscv.reg = %ub down to %lb step %step {
    riscv.addi %acc, 1 : (!riscv.reg<t0>) -> !riscv.reg<t0>
}
%i_last, %ub_last, %step_last = riscv_scf.while (%i0 = %lb, %ub_arg0 = %ub, %step_arg0 = %step) : (!riscv.reg, !riscv.reg, !riscv.reg) -> (!riscv.reg, !riscv.reg, !riscv.reg) {
        %cond = riscv.slt %i0, %ub_arg0 : (!riscv.reg, !riscv.reg) -> !riscv.reg
        riscv_scf.condition(%cond : !riscv.reg) %i0, %ub_arg0, %step_arg0 : !riscv.reg, !riscv.reg, !riscv.reg
} do {
    ^bb1(%i1 : !riscv.reg, %ub_arg1 : !riscv.reg, %step_arg1 : !riscv.reg):
        riscv.addi %acc, 1 : (!riscv.reg<t0>) -> !riscv.reg<t0>
        %i_next = "riscv.add"(%i1, %step_arg1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
        "riscv_scf.yield"(%i_next, %ub_arg1, %step_arg1) : (!riscv.reg, !riscv.reg, !riscv.reg) -> ()
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %lb = rv32.li 0 : !riscv.reg
// CHECK-NEXT:   %ub = rv32.li 100 : !riscv.reg
// CHECK-NEXT:   %step = rv32.li 1 : !riscv.reg
// CHECK-NEXT:   %acc = rv32.li 0 : !riscv.reg<t0>
// CHECK-NEXT:   riscv_scf.for %i : !riscv.reg = %lb to %ub step %step {
// CHECK-NEXT:     %0 = riscv.addi %acc, 1 : (!riscv.reg<t0>) -> !riscv.reg<t0>
// CHECK-NEXT:   }
// CHECK-NEXT:   riscv_scf.rof %j : !riscv.reg = %ub down to %lb step %step {
// CHECK-NEXT:     %1 = riscv.addi %acc, 1 : (!riscv.reg<t0>) -> !riscv.reg<t0>
// CHECK-NEXT:   }
// CHECK-NEXT:     %i_last, %ub_last, %step_last = riscv_scf.while (%i0 = %lb, %ub_arg0 = %ub, %step_arg0 = %step) : (!riscv.reg, !riscv.reg, !riscv.reg) -> (!riscv.reg, !riscv.reg, !riscv.reg) {
// CHECK-NEXT:             %cond = riscv.slt %i0, %ub_arg0 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:             riscv_scf.condition(%cond : !riscv.reg) %i0, %ub_arg0, %step_arg0 : !riscv.reg, !riscv.reg, !riscv.reg
// CHECK-NEXT:     } do {
// CHECK-NEXT:         ^bb0(%i1 : !riscv.reg, %ub_arg1 : !riscv.reg, %step_arg1 : !riscv.reg):
// CHECK-NEXT:             riscv.addi %acc, 1 : (!riscv.reg<t0>) -> !riscv.reg<t0>
// CHECK-NEXT:             %i_next = riscv.add %i1, %step_arg1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:             riscv_scf.yield %i_next, %ub_arg1, %step_arg1 : !riscv.reg, !riscv.reg, !riscv.reg
// CHECK-NEXT:     }
// CHECK-NEXT: }
