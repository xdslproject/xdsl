// RUN: xdsl-opt --split-input-file --verify-diagnostics %s | filecheck %s --match-full-lines

// Valid register names as sanity check

"test.op"() : () -> (!riscv.reg<a0>)
"test.op"() : () -> (!riscv.reg<j_0>)
"test.op"() : () -> (!riscv.freg<ft0>)
"test.op"() : () -> (!riscv.freg<fj_0>)

//      CHECK:  %0 = "test.op"() : () -> !riscv.reg<a0>
// CHECK-NEXT:  %1 = "test.op"() : () -> !riscv.reg<j_0>
// CHECK-NEXT:  %2 = "test.op"() : () -> !riscv.freg<ft0>
// CHECK-NEXT:  %3 = "test.op"() : () -> !riscv.freg<fj_0>

// -----

// Invalid integer register name
"test.op"() : () -> (!riscv.reg<ft0>)

//      CHECK:  Invalid register name ft0 for register set RV32I.

// -----

// Invalid float register name
"test.op"() : () -> (!riscv.freg<a0>)

//      CHECK:  Invalid register name a0 for register set RV32F.

// -----

// Non-existent integer register name
"test.op"() : () -> (!riscv.reg<x99>)

//      CHECK:  Invalid register name x99 for register set RV32I.

// -----

// Non-existent float register name
"test.op"() : () -> (!riscv.freg<ft99>)

//      CHECK:  Invalid register name ft99 for register set RV32F.
