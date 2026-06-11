// RUN: xdsl-opt --parsing-diagnostics --verify-diagnostics --split-input-file %s | filecheck %s

%lb = "rv32.li"() {"immediate" = 0: i32} : () -> !riscv.reg
%ub = "rv32.li"() {"immediate" = 100: i32} : () -> !riscv.reg
%step = "rv32.li"() {"immediate" = 1: i32} : () -> !riscv.reg
%acc = "rv32.li"() {"immediate" = 0 : i32} : () -> !riscv.reg<t0>

%i_last, %ub_last, %step_last = riscv_scf.while (%i0 = %lb, %step_arg0 = %step) : (!riscv.reg, !riscv.reg, !riscv.reg) -> (!riscv.reg, !riscv.reg, !riscv.reg) {
    %cond = riscv.slt %i0, %ub_arg0 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    riscv_scf.condition(%cond : !riscv.reg) %i0, %ub_arg0, %step_arg0 : !riscv.reg, !riscv.reg, !riscv.reg
} do {
^bb1(%i1: !riscv.reg, %ub_arg1: !riscv.reg, %step_arg1: !riscv.reg):
    "riscv.addi"(%acc) {"immediate" = 1 : i12} : (!riscv.reg<t0>) -> !riscv.reg<t0>
    %i_next = "riscv.add"(%i1, %step_arg1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
    "riscv_scf.yield"(%i_next, %ub_arg1, %step_arg1) : (!riscv.reg, !riscv.reg, !riscv.reg) -> ()
}

// CHECK: Mismatch between block argument count (2) and operand count (3)

// -----

%lb, %ub, %step = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)

"riscv_scf.for"(%lb, %ub, %step) <{step_attr = 1 : si12, operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
^bb0(%i: !riscv.reg):
    yield
}) : (!riscv.reg, !riscv.reg, !riscv.reg) -> ()

// CHECK: Operation does not verify: Exactly one of step_attr (static) or step_val (dynamic) must be set, got step_attr=1 : si12, step_val=<OpResult[!riscv.reg] name_hint: step, index: 2, operation: test.op, uses: 1>

// -----

%lb, %ub = "test.op"() : () -> (!riscv.reg, !riscv.reg)

riscv_scf.for %i_static : !riscv.reg = %lb to %ub step 1 : f32 {
    yield
}

// CHECK: Expected IntegerAttr
