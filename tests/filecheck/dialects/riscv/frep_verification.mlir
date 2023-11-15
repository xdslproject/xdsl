// RUN: xdsl-opt --split-input-file --verify-diagnostics %s | filecheck %s

%i0 = "test.op"() : () -> !riscv.reg<a0>
%ft0, %ft1 = "test.op"() : () -> (!riscv.freg<ft0>, !riscv.freg<ft1>)
riscv_snitch.frep_outer %i0, 1, 0 {
}

// CHECK: expected a single block, but got 0 blocks

// -----

%i0 = "test.op"() : () -> !riscv.reg<a0>
%ft0, %ft1 = "test.op"() : () -> (!riscv.freg<ft0>, !riscv.freg<ft1>)
riscv_snitch.frep_outer %i0, 1, 0 {
^bb0:
}

// CHECK: Operation does not verify: Non-zero stagger mask currently unsupported

// -----

%i0 = "test.op"() : () -> !riscv.reg<a0>
%ft0, %ft1 = "test.op"() : () -> (!riscv.freg<ft0>, !riscv.freg<ft1>)
riscv_snitch.frep_outer %i0, 0, 1 {
^bb0:
}

// CHECK: Operation does not verify: Non-zero stagger count currently unsupported

// -----

%i0 = "test.op"() : () -> !riscv.reg<a0>
%ft0, %ft1 = "test.op"() : () -> (!riscv.freg<ft0>, !riscv.freg<ft1>)
riscv_snitch.frep_outer %i0 {
    riscv.sw %i0, %i0, 0 : (!riscv.reg<a0>, !riscv.reg<a0>) -> ()
}

// CHECK: Operation does not verify: Frep operation body may not contain instructions with side-effects, found riscv.sw
