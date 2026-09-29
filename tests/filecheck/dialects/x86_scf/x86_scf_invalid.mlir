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

%lb, %ub, %step = "test.op"() : () -> (!x86.reg64, !x86.reg64, !x86.reg64)

"x86_scf.for"(%lb, %ub, %step) <{ub_attr = 10 : si32, operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
^bb0(%i: !x86.reg64):
    x86_scf.yield
}) : (!x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64)

// CHECK: Operation does not verify: Exactly one of ub_attr (static) or ub_val (dynamic) must be set

// -----

%lb, %ub, %step = "test.op"() : () -> (!x86.reg64, !x86.reg64, !x86.reg64)

"x86_scf.for"(%lb, %ub, %step) <{step_attr = 1 : si32, operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
^bb0(%i: !x86.reg64):
    x86_scf.yield
}) : (!x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64)

// CHECK: Operation does not verify: Exactly one of step_attr (static) or step_val (dynamic) must be set

// -----

%lb = "test.op"() : () -> !x86.reg64

x86_scf.for %i : !x86.reg64 = %lb to 1 : f32 step 1 : si32 {
    x86_scf.yield
}

// CHECK: Expected IntegerAttr

// -----

%lb = "test.op"() : () -> !x86.reg64

x86_scf.for %i : !x86.reg64 = %lb to 1 : si32 step 1 : f32 {
    x86_scf.yield
}

// CHECK: Expected IntegerAttr

// -----

%lb = "test.op"() : () -> !x86.reg64

x86_scf.for %i : !x86.reg64 = %lb to 1 : i64 step 1 : si32 {
    x86_scf.yield
}

// CHECK: Expected attribute si32 but got i64

// -----

%lb = "test.op"() : () -> !x86.reg64

x86_scf.for %i : !x86.reg64 = %lb to 1 : si32 step 1 : i64 {
    x86_scf.yield
}

// CHECK: Expected attribute si32 but got i64

// -----

%ub = "test.op"() : () -> !x86.reg64

x86_scf.rof %i : !x86.reg64 = %ub down to 0 : si32 step 1 : si32 {
    x86_scf.yield
}

// CHECK: Expected an operand.

// -----

%lb = "test.op"() : () -> !x86.reg64<rax>

"x86_scf.for"(%lb) <{ub_attr = 10 : si32, step_attr = 1 : si32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> ({
^bb0(%i: !x86.reg64<rbx>):
    x86_scf.yield
}) : (!x86.reg64<rax>) -> (!x86.reg64<rax>)

// CHECK: Operation does not verify: Expected induction var to be same type as lb

// -----

%lb, %ub, %init = "test.op"() : () -> (!x86.reg64, !x86.reg64, !x86.reg64<r11>)

%lb_end, %result = x86_scf.for %i : !x86.reg64 = %lb to %ub step 1 : si32 iter_args(%acc = %init) -> (!x86.reg64<r11>) {
    x86_scf.yield
}

// CHECK: Expected 1 args, got 0. The x86_scf.for must yield its carried variables.

// -----

%lb, %ub, %init = "test.op"() : () -> (!x86.reg64, !x86.reg64, !x86.reg64<r11>)
%wrong = "test.op"() : () -> !x86.reg64<r10>

%lb_end, %result = x86_scf.for %i : !x86.reg64 = %lb to %ub step 1 : si32 iter_args(%acc = %init) -> (!x86.reg64<r11>) {
    x86_scf.yield %wrong : !x86.reg64<r10>
}

// CHECK: Expected !x86.reg64<r11>, got !x86.reg64<r10>. The x86_scf.for's x86_scf.yield must match carried variables types.

// -----

%lb, %ub, %init = "test.op"() : () -> (!x86.reg64, !x86.reg64, !x86.reg64<r11>)

%lb_end, %result = x86_scf.rof %i : !x86.reg64 = %ub down to %lb step 1 : si32 iter_args(%acc = %init) -> (!x86.reg64<r11>) {
    x86_scf.yield
}

// CHECK: Expected 1 args, got 0. The x86_scf.rof must yield its carried variables.

// -----

%lb, %ub, %init = "test.op"() : () -> (!x86.reg64, !x86.reg64, !x86.reg64<r11>)
%wrong = "test.op"() : () -> !x86.reg64<r10>

%lb_end, %result = x86_scf.rof %i : !x86.reg64 = %ub down to %lb step 1 : si32 iter_args(%acc = %init) -> (!x86.reg64<r11>) {
    x86_scf.yield %wrong : !x86.reg64<r10>
}

// CHECK: Expected !x86.reg64<r11>, got !x86.reg64<r10>. The x86_scf.rof's x86_scf.yield must match carried variables types.
