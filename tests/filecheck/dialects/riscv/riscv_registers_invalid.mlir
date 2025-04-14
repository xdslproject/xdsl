// RUN: xdsl-opt --split-input-file --verify-diagnostics %s | filecheck %s

// Valid register names as sanity check

"test.op"() : () -> (!riscv.reg<a0>)
"test.op"() : () -> (!riscv.reg<x0>)
"test.op"() : () -> (!riscv.reg<j_0>)
"test.op"() : () -> (!riscv.freg<ft0>)
"test.op"() : () -> (!riscv.freg<f0>)
"test.op"() : () -> (!riscv.freg<fj_0>)

//      CHECK:  "test.op"() : () -> !riscv.reg<a0>
// CHECK-NEXT:  "test.op"() : () -> !riscv.reg<x0>
// CHECK-NEXT:  "test.op"() : () -> !riscv.reg<j_0>
// CHECK-NEXT:  "test.op"() : () -> !riscv.freg<ft0>
// CHECK-NEXT:  "test.op"() : () -> !riscv.freg<f0>
// CHECK-NEXT:  "test.op"() : () -> !riscv.freg<fj_0>

// -----

// Invalid integer register name
"test.op"() : () -> (!riscv.reg<ft0>)

//      CHECK:  Invalid register name ft0 for register type riscv.reg.

// -----

// Invalid float register name
"test.op"() : () -> (!riscv.freg<a0>)

//      CHECK:  Invalid register name a0 for register type riscv.freg.

// -----

// Non-existent integer register name
"test.op"() : () -> (!riscv.reg<x99>)

//      CHECK:  Invalid register name x99 for register type riscv.reg.

// -----

// Non-existent float register name
"test.op"() : () -> (!riscv.freg<ft99>)

//      CHECK:  Invalid register name ft99 for register type riscv.freg.

// -----

// Integer register with non-integer suffix
"test.op"() : () -> (!riscv.reg<j_j>)

//      CHECK:  Invalid register name j_j for register type riscv.reg.

// -----

// Float register with non-integer suffix
"test.op"() : () -> (!riscv.freg<fj_bla>)

//      CHECK:  Invalid register name fj_bla for register type riscv.freg.
