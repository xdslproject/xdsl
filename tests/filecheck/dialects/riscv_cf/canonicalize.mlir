// RUN: xdsl-opt -p canonicalize %s | filecheck %s

riscv_func.func @at_least_once() {
    %one = rv32.li 1 : !riscv.reg
    %three = rv32.li 3 : !riscv.reg
    %0 = riscv.mv %one : (!riscv.reg) -> !riscv.reg
    riscv_cf.bge %0 : !riscv.reg, %three : !riscv.reg, ^bb1(%0 : !riscv.reg), ^bb0(%0 : !riscv.reg)
^bb0(%i : !riscv.reg):
    riscv.label "scf_body_0_for"
    "test.op"(%i) : (!riscv.reg) -> ()
    %1 = riscv.add %i, %one : (!riscv.reg, !riscv.reg) -> !riscv.reg
    riscv_cf.blt %1 : !riscv.reg, %three : !riscv.reg, ^bb0(%1 : !riscv.reg), ^bb1(%1 : !riscv.reg)
^bb1(%2 : !riscv.reg):
    riscv.label "scf_body_end_0_for"
    riscv_func.return
}
riscv_func.func @never() {
    %one = rv32.li 1 : !riscv.reg
    %three = rv32.li 3 : !riscv.reg
    %0 = riscv.mv %one : (!riscv.reg) -> !riscv.reg
    riscv_cf.bge %three : !riscv.reg, %one : !riscv.reg, ^bb1(%0 : !riscv.reg), ^bb0(%0 : !riscv.reg)
^bb0(%i : !riscv.reg):
    riscv.label "scf_body_0_for"
    "test.op"(%i) : (!riscv.reg) -> ()
    %1 = riscv.add %i, %one : (!riscv.reg, !riscv.reg) -> !riscv.reg
    riscv_cf.blt %1 : !riscv.reg, %three : !riscv.reg, ^bb0(%1 : !riscv.reg), ^bb1(%1 : !riscv.reg)
^bb1(%2 : !riscv.reg):
    riscv.label "scf_body_end_0_for"
    riscv_func.return
}

// CHECK:       riscv_func.func @at_least_once() {
// CHECK-NEXT:    %one = rv32.li 1 : !riscv.reg
// CHECK-NEXT:    %three = rv32.li 3 : !riscv.reg
// CHECK-NEXT:    %0 = riscv.mv %one : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:    riscv_cf.branch ^bb0(%0 : !riscv.reg) attributes {comment = "Constant folded riscv_cf.bge"}
// CHECK-NEXT:  ^bb0(%i : !riscv.reg):
// CHECK-NEXT:    riscv.label "scf_body_0_for"
// CHECK-NEXT:    "test.op"(%i) : (!riscv.reg) -> ()
// CHECK-NEXT:    %1 = riscv.addi %i, 1 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:    riscv_cf.blt %1 : !riscv.reg, %three : !riscv.reg, ^bb0(%1 : !riscv.reg), ^bb1(%1 : !riscv.reg)
// CHECK-NEXT:  ^bb1(%2 : !riscv.reg):
// CHECK-NEXT:    riscv.label "scf_body_end_0_for"
// CHECK-NEXT:    riscv_func.return
// CHECK-NEXT:  }

// CHECK-NEXT:  riscv_func.func @never() {
// CHECK-NEXT:    %one = rv32.li 1 : !riscv.reg
// CHECK-NEXT:    %0 = riscv.mv %one : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:    riscv_cf.j ^bb0(%0 : !riscv.reg) attributes {comment = "Constant folded riscv_cf.bge"}
// CHECK-NEXT:  ^bb0(%1 : !riscv.reg):
// CHECK-NEXT:    riscv.label "scf_body_end_0_for"
// CHECK-NEXT:    riscv_func.return
// CHECK-NEXT:  }
