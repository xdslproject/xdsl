// RUN: xdsl-opt -p rvscf-allocate-registers --split-input-file %s | filecheck %s

builtin.module {   
  "riscv_func.func"() ({
  ^0(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>):
    %2 = riscv.mv %0 : (!riscv.reg<a0>) -> !riscv.reg<>
    %3 = riscv.mv %1 : (!riscv.reg<a1>) -> !riscv.reg<>
    %4 = riscv.li 1 : () -> !riscv.reg<>
    %5 = riscv.li 0 : () -> !riscv.reg<>
    %6 = "riscv_scf.for"(%2, %3, %4, %5) ({
    ^1(%7 : !riscv.reg<>, %8 : !riscv.reg<>):
      %9 = riscv.add %8, %7 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
      "riscv_scf.yield"(%9) : (!riscv.reg<>) -> ()
    }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %10 = riscv.mv %6 : (!riscv.reg<>) -> !riscv.reg<a0>
    "riscv_func.return"(%10) : (!riscv.reg<a0>) -> ()
  }) {"sym_name" = "foo"} : () -> ()
}

// CHECK:       builtin.module {
// CHECK-NEXT:  "riscv_func.func"() ({
// CHECK-NEXT:  ^0(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>):
// CHECK-NEXT:    %2 = riscv.mv %0 : (!riscv.reg<a0>) -> !riscv.reg<a0>
// CHECK-NEXT:    %3 = riscv.mv %1 : (!riscv.reg<a1>) -> !riscv.reg<a1>
// CHECK-NEXT:    %4 = riscv.li 1 : () -> !riscv.reg<a2>
// CHECK-NEXT:    %5 = riscv.li 0 : () -> !riscv.reg<a3>
// CHECK-NEXT:    %6 = "riscv_scf.for"(%2, %3, %4, %5) ({
// CHECK-NEXT:    ^1(%7 : !riscv.reg<a0>, %8 : !riscv.reg<a3>):
// CHECK-NEXT:      %9 = riscv.add %8, %7 : (!riscv.reg<a3>, !riscv.reg<a0>) -> !riscv.reg<a3>
// CHECK-NEXT:      "riscv_scf.yield"(%9) : (!riscv.reg<a3>) -> ()
// CHECK-NEXT:    }) : (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>) -> !riscv.reg<a3>
// CHECK-NEXT:    %10 = riscv.mv %6 : (!riscv.reg<a3>) -> !riscv.reg<a0>
// CHECK-NEXT:    "riscv_func.return"(%10) : (!riscv.reg<a0>) -> ()
// CHECK-NEXT:  }) {"sym_name" = "foo"} : () -> ()
// CHECK-NEXT:  }


// -----

builtin.module {   
  "riscv_func.func"() ({
  ^0(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>):
    %2 = riscv.mv %0 : (!riscv.reg<a0>) -> !riscv.reg<>
    %3 = riscv.mv %1 : (!riscv.reg<a1>) -> !riscv.reg<>
    %4 = riscv.li 1 : () -> !riscv.reg<>
    %55 = riscv.li 0 : () -> !riscv.reg<>
    %5 = riscv.fcvt.s.w %55 : (!riscv.reg<>) -> !riscv.freg<>
    %6 = "riscv_scf.for"(%2, %3, %4, %5) ({
    ^1(%7 : !riscv.reg<>, %8 : !riscv.freg<>):
      %99 = riscv.fcvt.s.w %7 : (!riscv.reg<>) -> !riscv.freg<>
      %9 = riscv.fadd.s %8, %99 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
      "riscv_scf.yield"(%9) : (!riscv.freg<>) -> ()
    }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.freg<>) -> !riscv.freg<>
    %10 = riscv.fcvt.w.s %6 : (!riscv.freg<>) -> !riscv.reg<a0>
    "riscv_func.return"(%10) : (!riscv.reg<a0>) -> ()
  }) {"sym_name" = "foo"} : () -> ()
}

// -----

builtin.module {   
  "riscv_func.func"() ({
  ^0(%arg0 : !riscv.reg<a0>):
    %0 = riscv.li 0 : () -> !riscv.reg<>
    %2 = riscv.li 0 : () -> !riscv.reg<>
    %3 = riscv.li 1 : () -> !riscv.reg<>
    %5 = "riscv_scf.for"(%2, %arg0, %3, %0) ({
    ^1(%arg1 : !riscv.reg<>, %arg2 : !riscv.reg<>):
      %6 = riscv.li 0 : () -> !riscv.reg<>
      %7 = riscv.li 1 : () -> !riscv.reg<>
      %9 = "riscv_scf.for"(%6, %arg0, %7, %arg2) ({
      ^2(%arg3 : !riscv.reg<>, %arg4 : !riscv.reg<>):
        %10 = riscv.add %arg1, %arg3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %11 = riscv.add %arg4, %10 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "riscv_scf.yield"(%11) : (!riscv.reg<>) -> ()
      }) : (!riscv.reg<>, !riscv.reg<a0>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
      "riscv_scf.yield"(%9) : (!riscv.reg<>) -> ()
    }) : (!riscv.reg<>, !riscv.reg<a0>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    "riscv_func.return"(%5) : (!riscv.reg<>) -> ()
  }) {"sym_name" = "foo"} : () -> ()
}


// CHECK:       builtin.module {
// CHECK-NEXT:    "riscv_func.func"() ({
// CHECK-NEXT:    ^0(%arg0 : !riscv.reg<a0>):
// CHECK-NEXT:      %0 = riscv.li 0 : () -> !riscv.reg<a1>
// CHECK-NEXT:      %1 = riscv.li 0 : () -> !riscv.reg<a2>
// CHECK-NEXT:      %2 = riscv.li 1 : () -> !riscv.reg<a3>
// CHECK-NEXT:      %3 = "riscv_scf.for"(%1, %arg0, %2, %0) ({
// CHECK-NEXT:      ^1(%arg1 : !riscv.reg<a2>, %arg2 : !riscv.reg<a1>):
// CHECK-NEXT:        %4 = riscv.li 0 : () -> !riscv.reg<a4>
// CHECK-NEXT:        %5 = riscv.li 1 : () -> !riscv.reg<a5>
// CHECK-NEXT:        %6 = "riscv_scf.for"(%4, %arg0, %5, %arg2) ({
// CHECK-NEXT:        ^2(%arg3 : !riscv.reg<a4>, %arg4 : !riscv.reg<a1>):
// CHECK-NEXT:          %7 = riscv.add %arg1, %arg3 : (!riscv.reg<a2>, !riscv.reg<a4>) -> !riscv.reg<a0>
// CHECK-NEXT:          %8 = riscv.add %arg4, %7 : (!riscv.reg<a1>, !riscv.reg<a0>) -> !riscv.reg<a1>
// CHECK-NEXT:          "riscv_scf.yield"(%8) : (!riscv.reg<a1>) -> ()
// CHECK-NEXT:        }) : (!riscv.reg<a4>, !riscv.reg<a0>, !riscv.reg<a5>, !riscv.reg<a1>) -> !riscv.reg<a1>
// CHECK-NEXT:        "riscv_scf.yield"(%6) : (!riscv.reg<a1>) -> ()
// CHECK-NEXT:      }) : (!riscv.reg<a2>, !riscv.reg<a0>, !riscv.reg<a3>, !riscv.reg<a1>) -> !riscv.reg<a1>
// CHECK-NEXT:      "riscv_func.return"(%3) : (!riscv.reg<a1>) -> ()
// CHECK-NEXT:    }) {"sym_name" = "foo"} : () -> ()
// CHECK-NEXT:  }