// RUN: xdsl-opt -t riscv-asm %s | filecheck %s

// A test that verifies that we can emit the target assembly for Snitch, below are the
// versions of ssum (C=A+B where all have fixed size 128xf32) .

riscv_func.func @pres_1(%X : !riscv.reg<a0>, %Y : !riscv.reg<a1>, %Z : !riscv.reg<a2>) {
    %zero = riscv.get_register : !riscv.reg<zero>
    %i = riscv.mv %zero : (!riscv.reg<zero>) -> !riscv.reg<a3>
    %ub = riscv.addi %zero 512 : (!riscv.reg<zero>) -> !riscv.reg<a4>
    riscv.label ".loop_body"
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

riscv_func.func @pres_2(%X : !riscv.reg<a0>, %Y : !riscv.reg<a1>, %Z : !riscv.reg<a2>) {
    %zero = riscv.get_register : !riscv.reg<zero>
    %i = riscv.mv %zero : (!riscv.reg<zero>) -> !riscv.reg<a3>
    %ub = riscv.addi %zero 512 : (!riscv.reg<zero>) -> !riscv.reg<a4>
    riscv.label ".loop_body"
    %x_i = riscv.add %X, %i : (!riscv.reg<a0>, !riscv.reg<a3>) -> !riscv.reg<a5>
    %x = riscv.fld %x_i, 0 : (!riscv.reg<a5>) -> !riscv.freg<ft0>
    %y_i = riscv.add %Y, %i : (!riscv.reg<a1>, !riscv.reg<a3>) -> !riscv.reg<a5>
    %y = riscv.fld %y_i, 0 : (!riscv.reg<a5>) -> !riscv.freg<ft1>
    %z = riscv.vfadd.s %y, %x : (!riscv.freg<ft1>, !riscv.freg<ft0>) -> !riscv.freg<ft0>
    %z_i = riscv.add %Z, %i : (!riscv.reg<a2>, !riscv.reg<a3>) -> !riscv.reg<a5>
    %i_next = riscv.addi %i, 8 : (!riscv.reg<a3>) -> !riscv.reg<a3>
    riscv.fsd %z_i, %z, 0 : (!riscv.reg<a5>, !riscv.freg<ft0>) -> ()
    riscv.bne %i_next, %ub, ".loop_body" : (!riscv.reg<a3>, !riscv.reg<a4>) -> ()
    riscv_func.return
}

riscv_func.func @pres_3(%X : !riscv.reg<a0>, %Y : !riscv.reg<a1>, %Z : !riscv.reg<a2>) {
    %zero = riscv.get_register : !riscv.reg<zero>
    %n = riscv.addi %zero 63 : (!riscv.reg<zero>) -> !riscv.reg<a3>
    riscv_snitch.scfgwi %n, 95 : (!riscv.reg<a3>) -> ()
    %ub = riscv.addi %zero 8 : (!riscv.reg<zero>) -> !riscv.reg<a3>
    riscv_snitch.scfgwi %n, 223 : (!riscv.reg<a3>) -> ()
    riscv_snitch.scfgwi %X, 768 : (!riscv.reg<a0>) -> ()
    riscv_snitch.scfgwi %Y, 769 : (!riscv.reg<a1>) -> ()
    riscv_snitch.scfgwi %Z, 898 : (!riscv.reg<a2>) -> ()
    %zero_6 = riscv.csrrsi 1984, 1 : () -> !riscv.reg<zero>
    %i = riscv.addi %zero, 64 : (!riscv.reg<zero>) -> !riscv.reg<a0>
    riscv.label ".loop_body"
    %x = riscv.get_float_register : !riscv.freg<ft0>
    %y = riscv.get_float_register : !riscv.freg<ft1>
    %z = riscv.vfadd.s %x, %y : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
    %i_next = riscv.addi %i, -1 : (!riscv.reg<a0>) -> !riscv.reg<a0>
    riscv.bne %i_next, %zero, ".loop_body" : (!riscv.reg<a0>, !riscv.reg<zero>) -> ()
    %zero_7 = riscv.csrrci 1984, 1 : () -> !riscv.reg<zero>
    riscv_func.return
}

riscv_func.func @pres_4(%X : !riscv.reg<a0>, %Y : !riscv.reg<a1>, %Z : !riscv.reg<a2>) {
    %zero = riscv.get_register : !riscv.reg<zero>
    %n = riscv.addi %zero 63 : (!riscv.reg<zero>) -> !riscv.reg<a3>
    riscv_snitch.scfgwi %n, 95 : (!riscv.reg<a3>) -> ()
    %ub = riscv.addi %zero 8 : (!riscv.reg<zero>) -> !riscv.reg<a3>
    riscv_snitch.scfgwi %n, 223 : (!riscv.reg<a3>) -> ()
    riscv_snitch.scfgwi %X, 768 : (!riscv.reg<a0>) -> ()
    riscv_snitch.scfgwi %Y, 769 : (!riscv.reg<a1>) -> ()
    riscv_snitch.scfgwi %Z, 898 : (!riscv.reg<a2>) -> ()
    %zero_6 = riscv.csrrsi 1984, 1 : () -> !riscv.reg<zero>
    %i = riscv.addi %zero, 63 : (!riscv.reg<zero>) -> !riscv.reg<a0>
    %x = riscv.get_float_register : !riscv.freg<ft0>
    %y = riscv.get_float_register : !riscv.freg<ft1>
    riscv_snitch.frep_outer %i {
        %z = riscv.vfadd.s %x, %y : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
    }
    %zero_7 = riscv.csrrci 1984, 1 : () -> !riscv.reg<zero>
    riscv_func.return
}

// CHECK:        .text
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

// CHECK-NEXT:   pres_2:
// CHECK-NEXT:       mv  a3, zero
// CHECK-NEXT:       addi  a4, zero, 512
// CHECK-NEXT:  .loop_body:
// CHECK-NEXT:       add  a5, a0, a3
// CHECK-NEXT:       fld  ft0, 0(a5)
// CHECK-NEXT:       add  a5, a1, a3
// CHECK-NEXT:       fld  ft1, 0(a5)
// CHECK-NEXT:       vfadd.s  ft0, ft1, ft0
// CHECK-NEXT:       add  a5, a2, a3
// CHECK-NEXT:       addi  a3, a3, 8
// CHECK-NEXT:       fsd  ft0, 0(a5)
// CHECK-NEXT:       bne  a3, a4, .loop_body
// CHECK-NEXT:       ret

// CHECK-NEXT:   pres_3:
// CHECK-NEXT:     addi  a3, zero, 63
// CHECK-NEXT:     scfgwi  a3, 95
// CHECK-NEXT:     addi  a3, zero, 8
// CHECK-NEXT:     scfgwi  a3, 223
// CHECK-NEXT:     scfgwi  a0, 768
// CHECK-NEXT:     scfgwi  a1, 769
// CHECK-NEXT:     scfgwi  a2, 898
// CHECK-NEXT:     csrrsi  zero, 1984, 1
// CHECK-NEXT:     addi  a0, zero, 64
// CHECK-NEXT:   .loop_body:
// CHECK-NEXT:     vfadd.s  ft2, ft0, ft1
// CHECK-NEXT:     addi  a0, a0, -1
// CHECK-NEXT:     bne  a0, zero, .loop_body
// CHECK-NEXT:     csrrci  zero, 1984, 1
// CHECK-NEXT:     ret

// CHECK-NEXT:   pres_4:
// CHECK-NEXT:     addi  a3, zero, 63
// CHECK-NEXT:     scfgwi  a3, 95
// CHECK-NEXT:     addi  a3, zero, 8
// CHECK-NEXT:     scfgwi  a3, 223
// CHECK-NEXT:     scfgwi  a0, 768
// CHECK-NEXT:     scfgwi  a1, 769
// CHECK-NEXT:     scfgwi  a2, 898
// CHECK-NEXT:     csrrsi  zero, 1984, 1
// CHECK-NEXT:     addi  a0, zero, 63
// CHECK-NEXT:     frep.o a0, 1, 0, 0
// CHECK-NEXT:     vfadd.s  ft2, ft0, ft1
// CHECK-NEXT:     csrrci  zero, 1984, 1
// CHECK-NEXT:     ret
