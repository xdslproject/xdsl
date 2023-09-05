// RUN: XDSL_ROUNDTRIP
"builtin.module"() ({
    %lb = "riscv.li"() {"immediate" = 0: i32} : () -> !riscv.reg<>
    %ub = "riscv.li"() {"immediate" = 100: i32} : () -> !riscv.reg<>
    %step = "riscv.li"() {"immediate" = 1: i32} : () -> !riscv.reg<>
    %acc = "riscv.li"() {"immediate" = 0 : i32} : () -> !riscv.reg<t0>
    "riscv_scf.for"(%lb, %ub, %step) ({
        ^1(%i: !riscv.reg<>):
            "riscv.addi"(%acc) {"immediate" = 1 : i12} : (!riscv.reg<t0>) -> !riscv.reg<t0>
            "riscv_scf.yield"() : () -> ()
    }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
    %i_last, %ub_last, %step_last = "riscv_scf.while"(%lb, %ub, %step) ({
        ^1(%i0 : !riscv.reg<>, %ub_arg0 : !riscv.reg<>, %step_arg0 : !riscv.reg<>):
            %cond = riscv.slt %i0, %ub_arg0 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
            "riscv_scf.condition"(%cond, %i0, %ub_arg0, %step_arg0) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
    }, {
        ^2(%i1 : !riscv.reg<>, %ub_arg1 : !riscv.reg<>, %step_arg1 : !riscv.reg<>):
            "riscv.addi"(%acc) {"immediate" = 1 : i12} : (!riscv.reg<t0>) -> !riscv.reg<t0>
            %i_next = "riscv.add"(%i1, %step_arg1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
            "riscv_scf.yield"(%i_next, %ub_arg1, %step_arg1) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
    }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>)
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %lb = riscv.li 0 : () -> !riscv.reg<>
// CHECK-NEXT:   %ub = riscv.li 100 : () -> !riscv.reg<>
// CHECK-NEXT:   %step = riscv.li 1 : () -> !riscv.reg<>
// CHECK-NEXT:   %acc = riscv.li 0 : () -> !riscv.reg<t0>
// CHECK-NEXT:   "riscv_scf.for"(%lb, %ub, %step) ({
// CHECK-NEXT:   ^0(%i : !riscv.reg<>):
// CHECK-NEXT:     %0 = riscv.addi %acc, 1 : (!riscv.reg<t0>) -> !riscv.reg<t0>
// CHECK-NEXT:     "riscv_scf.yield"() : () -> ()
// CHECK-NEXT:   }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:     %i_last, %ub_last, %step_last = "riscv_scf.while"(%lb, %ub, %step) ({
// CHECK-NEXT:         ^1(%i0 : !riscv.reg<>, %ub_arg0 : !riscv.reg<>, %step_arg0 : !riscv.reg<>):
// CHECK-NEXT:             %cond = riscv.slt %i0, %ub_arg0 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:             "riscv_scf.condition"(%cond, %i0, %ub_arg0, %step_arg0) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:     }, {
// CHECK-NEXT:         ^2(%i1 : !riscv.reg<>, %ub_arg1 : !riscv.reg<>, %step_arg1 : !riscv.reg<>):
// CHECK-NEXT:             riscv.addi %acc, 1 : (!riscv.reg<t0>) -> !riscv.reg<t0>
// CHECK-NEXT:             %i_next = riscv.add %i1, %step_arg1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:             "riscv_scf.yield"(%i_next, %ub_arg1, %step_arg1) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:     }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>)
// CHECK-NEXT: }
