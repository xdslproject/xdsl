// RUN: xdsl-opt --split-input-file --verify-diagnostics %s | filecheck %s


%0, %init = "test.op"() : () -> (!riscv.reg<a0>, !riscv.freg<fa0>)

%z = "riscv_snitch.frep_outer"(%0, %init) ({
^0(%acc : !riscv.freg<fa0>):
    riscv.sw %0, %0, 0 : (!riscv.reg<a0>, !riscv.reg<a0>) -> ()
    %res = "riscv.fadd.d"(%acc, %acc) : (!riscv.freg<fa0>, !riscv.freg<fa0>) -> !riscv.freg<fa0>
    "riscv_snitch.frep_yield"(%res) : (!riscv.freg<fa0>) -> ()
}) {"stagger_mask" = #builtin.int<0>, "stagger_count" = #builtin.int<0>} : (!riscv.reg<a0>, !riscv.freg<fa0>) -> !riscv.freg<fa0>

// CHECK: Operation does not verify: Frep operation body may not contain instructions with side-effects, found riscv.sw

// -----

%0, %init = "test.op"() : () -> (!riscv.reg<a0>, !riscv.freg<fa0>)

%z = "riscv_snitch.frep_outer"(%0, %init) ({
^0(%acc : !riscv.freg<fa0>, %extra : !riscv.freg<fa1>):
    %res = "riscv.fadd.d"(%acc, %acc) : (!riscv.freg<fa0>, !riscv.freg<fa0>) -> !riscv.freg<fa0>
    "riscv_snitch.frep_yield"(%res) : (!riscv.freg<fa0>) -> ()
}) {"stagger_mask" = #builtin.int<0>, "stagger_count" = #builtin.int<0>} : (!riscv.reg<a0>, !riscv.freg<fa0>) -> !riscv.freg<fa0>

// CHECK: Operation does not verify: Wrong number of block arguments, expected 1, got 2. The body must have the induction variable and loop-carried variables as arguments.

// -----

%0, %init = "test.op"() : () -> (!riscv.reg<a0>, !riscv.freg<fa0>)

%z = "riscv_snitch.frep_outer"(%0, %init) ({
^0(%acc : !riscv.freg<fa1>):
    %res = "riscv.fadd.d"(%acc, %acc) : (!riscv.freg<fa1>, !riscv.freg<fa1>) -> !riscv.freg<fa0>
    "riscv_snitch.frep_yield"(%res) : (!riscv.freg<fa0>) -> ()
}) {"stagger_mask" = #builtin.int<0>, "stagger_count" = #builtin.int<0>} : (!riscv.reg<a0>, !riscv.freg<fa0>) -> !riscv.freg<fa0>

// CHECK: Operation does not verify: Block argument 0 has wrong type, expected !riscv.freg<fa0>, got !riscv.freg<fa1>. Arguments after the induction variable must match the carried variables.

// -----

%0, %init = "test.op"() : () -> (!riscv.reg<a0>, !riscv.freg<fa0>)

%z = "riscv_snitch.frep_outer"(%0, %init) ({
^0(%acc : !riscv.freg<fa0>):
    %res = "riscv.fadd.d"(%acc, %acc) : (!riscv.freg<fa0>, !riscv.freg<fa0>) -> !riscv.freg<fa0>
    "riscv_snitch.frep_yield"() : () -> ()
}) {"stagger_mask" = #builtin.int<0>, "stagger_count" = #builtin.int<0>} : (!riscv.reg<a0>, !riscv.freg<fa0>) -> !riscv.freg<fa0>

// CHECK: Operation does not verify: Expected 1 args, got 0. The riscv_scf.frep must yield its carried variables.

// -----

%0, %init = "test.op"() : () -> (!riscv.reg<a0>, !riscv.freg<fa0>)

%z = "riscv_snitch.frep_outer"(%0, %init) ({
^0(%acc : !riscv.freg<fa0>):
    %res = "riscv.fadd.d"(%acc, %acc) : (!riscv.freg<fa0>, !riscv.freg<fa0>) -> !riscv.freg<fa1>
    "riscv_snitch.frep_yield"(%res) : (!riscv.freg<fa1>) -> ()
}) {"stagger_mask" = #builtin.int<0>, "stagger_count" = #builtin.int<0>} : (!riscv.reg<a0>, !riscv.freg<fa0>) -> !riscv.freg<fa0>

// CHECK: Operation does not verify: Expected !riscv.freg<fa0>, got !riscv.freg<fa1>. The riscv_snitch.frep's riscv_snitch.frep_yield must match carriedvariables types.
