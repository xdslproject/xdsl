// RUN: xdsl-opt %s --parsing-diagnostics | filecheck %s



%lb = "rv32.li"() {"immediate" = 0: i32} : () -> !riscv.reg
%ub = "rv32.li"() {"immediate" = 100: i32} : () -> !riscv.reg
%step = "rv32.li"() {"immediate" = 1: i32} : () -> !riscv.reg
%acc = "rv32.li"() {"immediate" = 0 : i32} : () -> !riscv.reg<t0>

%i_last, %ub_last, %step_last = riscv_scf.while (%i0 = %lb, %step_arg0 = %step) : (!riscv.reg, !riscv.reg, !riscv.reg) -> (!riscv.reg, !riscv.reg, !riscv.reg) {
    %cond = riscv.slt %i0, %ub_arg0 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    riscv_scf.condition(%cond : !riscv.reg) %i0, %ub_arg0, %step_arg0 : !riscv.reg, !riscv.reg, !riscv.reg
} do {
^bb1(%i1 : !riscv.reg, %ub_arg1 : !riscv.reg, %step_arg1 : !riscv.reg):
    "riscv.addi"(%acc) {"immediate" = 1 : i12} : (!riscv.reg<t0>) -> !riscv.reg<t0>
    %i_next = "riscv.add"(%i1, %step_arg1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
    "riscv_scf.yield"(%i_next, %ub_arg1, %step_arg1) : (!riscv.reg, !riscv.reg, !riscv.reg) -> ()
}

// CHECK: Mismatch between block argument count (2) and operand count (3)
