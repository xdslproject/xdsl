// RUN: xdsl-opt --split-input-file --verify-diagnostics %s | filecheck %s

%i0 = "test.op"() : () -> !riscv.reg
"riscv_snitch.frep_outer"(%i0) ({
}) {"stagger_mask" = #builtin.int<0>, "stagger_count" = #builtin.int<0>} : (!riscv.reg) -> ()

// CHECK: expected a single block, but got 0 blocks

// -----

%i0 = "test.op"() : () -> !riscv.reg<a0>
%ft0, %ft1 = "test.op"() : () -> (!riscv.freg<ft0>, !riscv.freg<ft1>)
riscv_snitch.frep_outer %i0, 1, 0 {
}

// CHECK: Operation does not verify: Non-zero stagger mask currently unsupported

// -----

%i0 = "test.op"() : () -> !riscv.reg<a0>
%ft0, %ft1 = "test.op"() : () -> (!riscv.freg<ft0>, !riscv.freg<ft1>)
riscv_snitch.frep_outer %i0, 0, 1 {
}

// CHECK: Operation does not verify: Non-zero stagger count currently unsupported

// -----

%i0 = "test.op"() : () -> !riscv.reg<a0>
%ft0, %ft1 = "test.op"() : () -> (!riscv.freg<ft0>, !riscv.freg<ft1>)
riscv_snitch.frep_outer %i0 {
    riscv.sw %i0, %i0, 0 : (!riscv.reg<a0>, !riscv.reg<a0>) -> ()
}

// CHECK: Operation does not verify: Frep operation body may not contain instructions with side-effects, found riscv.sw

// -----

%0 = "test.op"(): () -> !riscv.reg

"riscv_snitch.frep_outer"(%0) ({
^bb0:
}) {"stagger_mask" = #builtin.int<0>, "stagger_count" = #builtin.int<0>} : (!riscv.reg) -> ()

// CHECK: Operation riscv_snitch.frep_outer contains empty block in single-block region that expects at least a terminator


// -----

%0 = "test.op"(): () -> !riscv.reg
%f0 = "test.op"(): () -> !riscv.freg

"riscv_snitch.frep_outer"(%0) ({
^bb0:
    %f1 = "riscv.fadd.s"(%f0, %f0) : (!riscv.freg, !riscv.freg) -> !riscv.freg
}) {"stagger_mask" = #builtin.int<0>, "stagger_count" = #builtin.int<0>} : (!riscv.reg) -> ()

// CHECK: Operation riscv.fadd.s terminates block in single-block region but is not a terminator


// -----

%0 = "test.op"(): () -> !riscv.reg
%f0 = "test.op"(): () -> !riscv.freg

"riscv_snitch.frep_outer"(%0) ({
^bb0:
    %f1 = "riscv.fadd.s"(%f0, %f0) : (!riscv.freg, !riscv.freg) -> !riscv.freg
    "test.termop"() : () -> ()
}) : (!riscv.reg) -> ()

// CHECK: Operation does not verify: 'riscv_snitch.frep_outer' terminates with operation test.termop instead of riscv_snitch.frep_yield
