// RUN: xdsl-opt -t riscv-asm %s | filecheck %s

// A test that verifies that we can emit the target assembly for Snitch, below are the
// versions of ssum (C=A+B where all have fixed size 128xf32) .

riscv.label ".text" : () -> ()

riscv_func.func @pres_1(%X : !riscv.reg<a0>, %Y : !riscv.reg<a1>, %Z : !riscv.reg<a2>) {
    %zero = "riscv.get_register"() : () -> !riscv.reg<zero>
    %i = riscv.mv %zero : (!riscv.reg<zero>) -> !riscv.reg<a3>
    %ub = riscv.addi %zero 512 : (!riscv.reg<zero>) -> !riscv.reg<a4>
    riscv.label ".loop_body" : () -> ()
    %x_i = riscv.add %X, %i : (!riscv.reg<a0>, !riscv.reg<a3>) -> !riscv.reg<a5>
    %x = riscv.flw %x_i, 0 : (!riscv.reg<a5>) -> !riscv.freg<ft0>
    %y_i = riscv.add %Y, %i : (!riscv.reg<a1>, !riscv.reg<a3>) -> !riscv.reg<a5>
    %y = riscv.flw %y_i, 0 : (!riscv.reg<a5>) -> !riscv.freg<ft1>
    %z = riscv.fadd.s %y, %x : (!riscv.freg<ft1>, !riscv.freg<ft0>) -> !riscv.freg<ft0>
    %z_i = riscv.add %Z, %i : (!riscv.reg<a2>, !riscv.reg<a3>) -> !riscv.reg<a5>
    %i_next = riscv.addi %i, 4 : (!riscv.reg<a3>) -> !riscv.reg<a3>
    riscv.fsw %z_i, %z, 0 : (!riscv.reg<a5>, !riscv.freg<ft0>) -> ()
    riscv.bne %i_next, %ub, ".loop_body" : (!riscv.reg<a3>, !riscv.reg<a4>) -> ()
    riscv_func.return
}


// CHECK:
// CHECK-NEXT:   pres_1:
// CHECK-NEXT:       mv a3, zero
// CHECK-NEXT:       addi a4, zero, 512
// CHECK-NEXT:   .loop_body:
// CHECK-NEXT:       add a5, a0, a3
// CHECK-NEXT:       flw ft0, 0(a5)
// CHECK-NEXT:       add a5, a1, a3
// CHECK-NEXT:       flw ft1, 0(a5)
// CHECK-NEXT:       fadd.s ft0, ft1, ft0
// CHECK-NEXT:       add a5, a2, a3
// CHECK-NEXT:       addi a3, a3, 4
// CHECK-NEXT:       fsw ft0, 0(a5)
// CHECK-NEXT:       bne a3, a4, .loop_body
// CHECK-NEXT:       ret
