// RUN:xdsl-opt %s -p 'apply-individual-rewrite{matched_operation_index=8 operation_name="riscv.add" pattern_name="AddImmediates"}'| filecheck %s

riscv_func.func @at_least_once() {
    %one = riscv.li 1 : () -> !riscv.reg<>
    %three = riscv.li 3 : () -> !riscv.reg<>
    %0 = riscv.mv %one : (!riscv.reg<>) -> !riscv.reg<>
    riscv_cf.bge %0 : !riscv.reg<>, %three : !riscv.reg<>, ^1(%0 : !riscv.reg<>), ^0(%0 : !riscv.reg<>)
^0(%i : !riscv.reg<>):
    riscv.label "scf_body_0_for"
    "test.op"(%i) : (!riscv.reg<>) -> ()
    %1 = riscv.add %i, %one : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    riscv_cf.blt %1 : !riscv.reg<>, %three : !riscv.reg<>, ^0(%1 : !riscv.reg<>), ^1(%1 : !riscv.reg<>)
^1(%2 : !riscv.reg<>):
    riscv.label "scf_body_end_0_for"
    riscv_func.return
}
riscv_func.func @never() {
    %one = riscv.li 1 : () -> !riscv.reg<>
    %three = riscv.li 3 : () -> !riscv.reg<>
    %0 = riscv.mv %one : (!riscv.reg<>) -> !riscv.reg<>
    riscv_cf.bge %three : !riscv.reg<>, %one : !riscv.reg<>, ^1(%0 : !riscv.reg<>), ^0(%0 : !riscv.reg<>)
^0(%i : !riscv.reg<>):
    riscv.label "scf_body_0_for"
    "test.op"(%i) : (!riscv.reg<>) -> ()
    %1 = riscv.add %i, %one : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    riscv_cf.blt %1 : !riscv.reg<>, %three : !riscv.reg<>, ^0(%1 : !riscv.reg<>), ^1(%1 : !riscv.reg<>)
^1(%2 : !riscv.reg<>):
    riscv.label "scf_body_end_0_for"
    riscv_func.return
}

//CHECK:         builtin.module {
// CHECK-NEXT:       riscv_func.func @at_least_once() {
// CHECK-NEXT:         %one = riscv.li 1 : () -> !riscv.reg<>
// CHECK-NEXT:         %three = riscv.li 3 : () -> !riscv.reg<>
// CHECK-NEXT:         %0 = riscv.mv %one : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:         riscv_cf.bge %0 : !riscv.reg<>, %three : !riscv.reg<>, ^0(%0 : !riscv.reg<>), ^1(%0 : !riscv.reg<>)
// CHECK-NEXT:       ^1(%i : !riscv.reg<>):
// CHECK-NEXT:         riscv.label "scf_body_0_for"
// CHECK-NEXT:         "test.op"(%i) : (!riscv.reg<>) -> ()
// CHECK-NEXT:         %1 = riscv.addi %i, 1 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:         riscv_cf.blt %1 : !riscv.reg<>, %three : !riscv.reg<>, ^1(%1 : !riscv.reg<>), ^0(%1 : !riscv.reg<>)
// CHECK-NEXT:       ^0(%2 : !riscv.reg<>):
// CHECK-NEXT:         riscv.label "scf_body_end_0_for"
// CHECK-NEXT:         riscv_func.return
// CHECK-NEXT:       }
// CHECK-NEXT:       riscv_func.func @never() {
// CHECK-NEXT:         %one_1 = riscv.li 1 : () -> !riscv.reg<>
// CHECK-NEXT:         %three_1 = riscv.li 3 : () -> !riscv.reg<>
// CHECK-NEXT:         %3 = riscv.mv %one_1 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:         riscv_cf.bge %three_1 : !riscv.reg<>, %one_1 : !riscv.reg<>, ^2(%3 : !riscv.reg<>), ^3(%3 : !riscv.reg<>)
// CHECK-NEXT:       ^3(%i_1 : !riscv.reg<>):
// CHECK-NEXT:         riscv.label "scf_body_0_for"
// CHECK-NEXT:         "test.op"(%i_1) : (!riscv.reg<>) -> ()
// CHECK-NEXT:         %4 = riscv.add %i_1, %one_1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:         riscv_cf.blt %4 : !riscv.reg<>, %three_1 : !riscv.reg<>, ^3(%4 : !riscv.reg<>), ^2(%4 : !riscv.reg<>)
// CHECK-NEXT:       ^2(%5 : !riscv.reg<>):
// CHECK-NEXT:         riscv.label "scf_body_end_0_for"
// CHECK-NEXT:         riscv_func.return
// CHECK-NEXT:       }
// CHECK-NEXT:     }
