// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | xdsl-opt | filecheck %s
"builtin.module"() ({
    %lb = "riscv.li"() {"immediate" = 0: i32} : () -> !riscv.reg<>
    %ub = "riscv.li"() {"immediate" = 100: i32} : () -> !riscv.reg<>
    %step = "riscv.li"() {"immediate" = 1: i32} : () -> !riscv.reg<>
    "riscv_scf.for"(%lb, %ub, %step) ({
        ^1(%i: !riscv.reg<>):
            "riscv_scf.yield"() : () -> ()
    }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %lb = riscv.li  {"immediate" = 0 : i32} : () -> !riscv.reg<>
// CHECK-NEXT:   %ub = riscv.li  {"immediate" = 100 : i32} : () -> !riscv.reg<>
// CHECK-NEXT:   %step = riscv.li  {"immediate" = 1 : i32} : () -> !riscv.reg<>
// CHECK-NEXT:   "riscv_scf.for"(%lb, %ub, %step) ({
// CHECK-NEXT:   ^0(%i : !riscv.reg<>):
// CHECK-NEXT:     "riscv_scf.yield"() : () -> ()
// CHECK-NEXT:   }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT: }
