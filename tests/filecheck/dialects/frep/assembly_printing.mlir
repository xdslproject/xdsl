// RUN: xdsl-opt -t riscv-asm %s | filecheck %s

%i0 = riscv.get_register : () -> !riscv.reg<a0>
%ft0 = riscv.get_float_register : () -> !riscv.freg<ft0>
%ft1 = riscv.get_float_register : () -> !riscv.freg<ft1>
riscv.frep_outer %i0, 0, 0 ({
    %ft2 = riscv.fadd.s %ft0, %ft1 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
}) : (!riscv.reg<a0>) -> ()
riscv.frep_inner %i0, 0, 0 ({
    %ft2 = riscv.fadd.s %ft0, %ft1 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
}) : (!riscv.reg<a0>) -> ()


// CHECK:          frep.outer a0, 1, 0, 0
// CHECK-NEXT:     fadd.s  ft2, ft0, ft1
// CHECK:          frep.inner a0, 1, 0, 0
// CHECK-NEXT:     fadd.s  ft2, ft0, ft1
