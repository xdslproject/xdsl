// RUN: xdsl-opt -p lower-riscv-scf-to-labels --split-input-file %s | filecheck %s

builtin.module {
    "riscv_func.func"() ({
    ^0(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>):
        %2 = riscv.mv %0 : (!riscv.reg<a0>) -> !riscv.reg<a0>
        %3 = riscv.mv %1 : (!riscv.reg<a1>) -> !riscv.reg<a1>
        %4 = riscv.li 1 : () -> !riscv.reg<a2>
        %5 = riscv.li 0 : () -> !riscv.reg<a3>
        %6 = riscv.fcvt.s.w %5 : (!riscv.reg<a3>) -> !riscv.freg<fa0>
        %7 = "riscv_scf.for"(%2, %3, %4, %6) ({
        ^1(%8 : !riscv.reg<a0>, %9 : !riscv.freg<fa0>):
        %10 = riscv.fcvt.s.w %8 : (!riscv.reg<a0>) -> !riscv.freg<fa1>
        %11 = riscv.fadd.s %9, %10 : (!riscv.freg<fa0>, !riscv.freg<fa1>) -> !riscv.freg<fa0>
        "riscv_scf.yield"(%11) : (!riscv.freg<fa0>) -> ()
        }) : (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.freg<fa0>) -> !riscv.freg<fa0>
        %12 = riscv.fcvt.w.s %7 : (!riscv.freg<fa0>) -> !riscv.reg<a0>
        "riscv_func.return"(%12) : (!riscv.reg<a0>) -> ()
    }) {"sym_name" = "foo"} : () -> ()
}

// CHECK:       builtin.module {
// CHECK-NEXT:    "riscv_func.func"() ({
// CHECK-NEXT:    ^0(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>):
// CHECK-NEXT:      %2 = riscv.mv %0 : (!riscv.reg<a0>) -> !riscv.reg<a0>
// CHECK-NEXT:      %3 = riscv.mv %1 : (!riscv.reg<a1>) -> !riscv.reg<a1>
// CHECK-NEXT:      %4 = riscv.li 1 : () -> !riscv.reg<a2>
// CHECK-NEXT:      %5 = riscv.li 0 : () -> !riscv.reg<a3>
// CHECK-NEXT:      %6 = riscv.fcvt.s.w %5 : (!riscv.reg<a3>) -> !riscv.freg<fa0>
// CHECK-NEXT:      %7 = riscv.get_register : () -> !riscv.reg<a0>
// CHECK-NEXT:      riscv.label {"label" = #riscv.label<"scf_cond_0_for">} ({
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      riscv.bge %7, %3, "scf_body_end_0_for" : (!riscv.reg<a0>, !riscv.reg<a1>) -> ()
// CHECK-NEXT:      riscv.label {"label" = #riscv.label<"scf_body_0_for">} ({
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      %8 = riscv.get_float_register : () -> !riscv.freg<fa0>
// CHECK-NEXT:      %9 = riscv.get_register : () -> !riscv.reg<a0>
// CHECK-NEXT:      %10 = riscv.fcvt.s.w %9 : (!riscv.reg<a0>) -> !riscv.freg<fa1>
// CHECK-NEXT:      %11 = riscv.fadd.s %8, %10 : (!riscv.freg<fa0>, !riscv.freg<fa1>) -> !riscv.freg<fa0>
// CHECK-NEXT:      %12 = riscv.add %7, %4 : (!riscv.reg<a0>, !riscv.reg<a2>) -> !riscv.reg<a0>
// CHECK-NEXT:      riscv.blt %7, %3, "scf_body_0_for" : (!riscv.reg<a0>, !riscv.reg<a1>) -> ()
// CHECK-NEXT:      riscv.label {"label" = #riscv.label<"scf_body_end_0_for">} ({
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      %13 = riscv.get_float_register : () -> !riscv.freg<fa0>
// CHECK-NEXT:      %14 = riscv.fcvt.w.s %13 : (!riscv.freg<fa0>) -> !riscv.reg<a0>
// CHECK-NEXT:      "riscv_func.return"(%14) : (!riscv.reg<a0>) -> ()
// CHECK-NEXT:    }) {"sym_name" = "foo"} : () -> ()
// CHECK-NEXT:  }

// -----

builtin.module {
    "riscv_func.func"() ({
    ^0(%arg0 : !riscv.reg<a0>):
        %0 = riscv.li 0 : () -> !riscv.reg<a1>
        %1 = riscv.li 0 : () -> !riscv.reg<a2>
        %2 = riscv.li 1 : () -> !riscv.reg<a3>
        %3 = "riscv_scf.for"(%1, %arg0, %2, %0) ({
        ^1(%arg1 : !riscv.reg<a2>, %arg2 : !riscv.reg<a1>):
        %4 = riscv.li 0 : () -> !riscv.reg<a4>
        %5 = riscv.li 1 : () -> !riscv.reg<a5>
        %6 = "riscv_scf.for"(%4, %arg0, %5, %arg2) ({
        ^2(%arg3 : !riscv.reg<a4>, %arg4 : !riscv.reg<a1>):
            %7 = riscv.add %arg1, %arg3 : (!riscv.reg<a2>, !riscv.reg<a4>) -> !riscv.reg<a0>
            %8 = riscv.add %arg4, %7 : (!riscv.reg<a1>, !riscv.reg<a0>) -> !riscv.reg<a1>
            "riscv_scf.yield"(%8) : (!riscv.reg<a1>) -> ()
        }) : (!riscv.reg<a4>, !riscv.reg<a0>, !riscv.reg<a5>, !riscv.reg<a1>) -> !riscv.reg<a1>
        "riscv_scf.yield"(%6) : (!riscv.reg<a1>) -> ()
        }) : (!riscv.reg<a2>, !riscv.reg<a0>, !riscv.reg<a3>, !riscv.reg<a1>) -> !riscv.reg<a1>
        "riscv_func.return"(%3) : (!riscv.reg<a1>) -> ()
    }) {"sym_name" = "foo"} : () -> ()
}

// CHECK:       builtin.module {
// CHECK-NEXT:    "riscv_func.func"() ({
// CHECK-NEXT:    ^0(%arg0 : !riscv.reg<a0>):
// CHECK-NEXT:      %0 = riscv.li 0 : () -> !riscv.reg<a1>
// CHECK-NEXT:      %1 = riscv.li 0 : () -> !riscv.reg<a2>
// CHECK-NEXT:      %2 = riscv.li 1 : () -> !riscv.reg<a3>
// CHECK-NEXT:      %3 = riscv.get_register : () -> !riscv.reg<a2>
// CHECK-NEXT:      riscv.label {"label" = #riscv.label<"scf_cond_0_for">} ({
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      riscv.bge %3, %arg0, "scf_body_end_0_for" : (!riscv.reg<a2>, !riscv.reg<a0>) -> ()
// CHECK-NEXT:      riscv.label {"label" = #riscv.label<"scf_body_0_for">} ({
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      %arg2 = riscv.get_register : () -> !riscv.reg<a1>
// CHECK-NEXT:      %arg1 = riscv.get_register : () -> !riscv.reg<a2>
// CHECK-NEXT:      %4 = riscv.li 0 : () -> !riscv.reg<a4>
// CHECK-NEXT:      %5 = riscv.li 1 : () -> !riscv.reg<a5>
// CHECK-NEXT:      %6 = riscv.get_register : () -> !riscv.reg<a4>
// CHECK-NEXT:      riscv.label {"label" = #riscv.label<"scf_cond_1_for">} ({
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      riscv.bge %6, %arg0, "scf_body_end_1_for" : (!riscv.reg<a4>, !riscv.reg<a0>) -> ()
// CHECK-NEXT:      riscv.label {"label" = #riscv.label<"scf_body_1_for">} ({
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      %arg4 = riscv.get_register : () -> !riscv.reg<a1>
// CHECK-NEXT:      %arg3 = riscv.get_register : () -> !riscv.reg<a4>
// CHECK-NEXT:      %7 = riscv.add %arg1, %arg3 : (!riscv.reg<a2>, !riscv.reg<a4>) -> !riscv.reg<a0>
// CHECK-NEXT:      %8 = riscv.add %arg4, %7 : (!riscv.reg<a1>, !riscv.reg<a0>) -> !riscv.reg<a1>
// CHECK-NEXT:      %9 = riscv.add %6, %5 : (!riscv.reg<a4>, !riscv.reg<a5>) -> !riscv.reg<a4>
// CHECK-NEXT:      riscv.blt %6, %arg0, "scf_body_1_for" : (!riscv.reg<a4>, !riscv.reg<a0>) -> ()
// CHECK-NEXT:      riscv.label {"label" = #riscv.label<"scf_body_end_1_for">} ({
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      %10 = riscv.get_register : () -> !riscv.reg<a1>
// CHECK-NEXT:      %11 = riscv.add %3, %2 : (!riscv.reg<a2>, !riscv.reg<a3>) -> !riscv.reg<a2>
// CHECK-NEXT:      riscv.blt %3, %arg0, "scf_body_0_for" : (!riscv.reg<a2>, !riscv.reg<a0>) -> ()
// CHECK-NEXT:      riscv.label {"label" = #riscv.label<"scf_body_end_0_for">} ({
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      %12 = riscv.get_register : () -> !riscv.reg<a1>
// CHECK-NEXT:      "riscv_func.return"(%12) : (!riscv.reg<a1>) -> ()
// CHECK-NEXT:    }) {"sym_name" = "foo"} : () -> ()
// CHECK-NEXT:  }