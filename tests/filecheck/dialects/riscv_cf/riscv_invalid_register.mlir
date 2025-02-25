// RUN: xdsl-opt --split-input-file --parsing-diagnostics --verify-diagnostics %s | filecheck %s --strict-whitespace --match-full-lines

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

//      CHECK:"test.op"() : () -> (!riscv.reg<ft0>)
// CHECK-NEXT:                                ^^^
// CHECK-NEXT:                                Invalid register spelling ft0 for class IntRegisterType

// -----

// Invalid float register name
"test.op"() : () -> (!riscv.freg<a0>)

//      CHECK:"test.op"() : () -> (!riscv.freg<a0>)
// CHECK-NEXT:                                 ^^
// CHECK-NEXT:                                 Invalid register spelling a0 for class FloatRegisterType

// -----

// Non-existent integer register name
"test.op"() : () -> (!riscv.reg<x99>)

//      CHECK:"test.op"() : () -> (!riscv.reg<x99>)
// CHECK-NEXT:                                ^^^
// CHECK-NEXT:                                Invalid register spelling x99 for class IntRegisterType

// -----

// Non-existent float register name
"test.op"() : () -> (!riscv.freg<ft99>)

//      CHECK:"test.op"() : () -> (!riscv.freg<ft99>)
// CHECK-NEXT:                                 ^^^^
// CHECK-NEXT:                                 Invalid register spelling ft99 for class FloatRegisterType

// -----

// Integer register with non-integer suffix
"test.op"() : () -> (!riscv.reg<j_j>)

//      CHECK:"test.op"() : () -> (!riscv.reg<j_j>)
// CHECK-NEXT:                                ^^^
// CHECK-NEXT:                                Invalid register spelling j_j for class IntRegisterType

// -----

// Float register with non-integer suffix
"test.op"() : () -> (!riscv.freg<fj_bla>)

//      CHECK:"test.op"() : () -> (!riscv.freg<fj_bla>)
// CHECK-NEXT:                                 ^^^^^
// CHECK-NEXT:                                 Invalid register spelling fj_bla for class FloatRegisterType
