// RUN: xdsl-opt -p riscv-scf-loop-fusion %s | filecheck %s

%c0 = riscv.li 0 : () -> !riscv.reg<>
%c1 = riscv.li 1 : () -> !riscv.reg<>
%c8 = riscv.li 8 : () -> !riscv.reg<>
%c64 = riscv.li 64 : () -> !riscv.reg<>

riscv_scf.for %16 : !riscv.reg<> = %c0 to %c64 step %c8 {
    riscv_scf.for %17 : !riscv.reg<> = %c0 to %c8 step %c1 {
    %18 = riscv.li 8 : () -> !riscv.reg<>
    %19 = riscv.add %16, %17 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    "test.op"(%19) : (!riscv.reg<>) -> ()
    }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %c0 = riscv.li 0 : () -> !riscv.reg<>
// CHECK-NEXT:    %c1 = riscv.li 1 : () -> !riscv.reg<>
// CHECK-NEXT:    %c8 = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:    %c64 = riscv.li 64 : () -> !riscv.reg<>
// CHECK-NEXT:    riscv_scf.for %0 : !riscv.reg<> = %c0 to %c64 step %c1 {
// CHECK-NEXT:      %1 = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:      "test.op"(%0) : (!riscv.reg<>) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:  }


