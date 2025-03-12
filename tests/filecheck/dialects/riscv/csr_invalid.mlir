// RUN: xdsl-opt --split-input-file --verify-diagnostics --parsing-diagnostics %s | filecheck %s

// Valid CSR operation with x0
%x0 = "test.op"() : () -> !riscv.reg<x0>
%0 = riscv.csrrw %x0, 16 : (!riscv.reg<x0>) -> !riscv.reg

// Valid CSR operation with zero
%zero = "test.op"() : () -> !riscv.reg<zero>
%1 = riscv.csrrw %zero, 16 : (!riscv.reg<zero>) -> !riscv.reg

// -----

// Invalid CSR operation with a1
%a1 = "test.op"() : () -> !riscv.reg<a1>
%2 = riscv.csrrw %a1, 16, "w" : (!riscv.reg<a1>) -> !riscv.reg<a1>

// CHECK: Operation does not verify: When in 'writeonly' mode, destination must be register x0 (a.k.a. 'zero'), not 'a1'

// -----

// Invalid CSR operation with t0
%t0 = "test.op"() : () -> !riscv.reg<t0>
%3 = riscv.csrrs %t0, 16, "r" : (!riscv.reg<t0>) -> !riscv.reg

// CHECK: Operation does not verify: When in 'readonly' mode, source must be register x0 (a.k.a. 'zero'), not 't0'

// -----

// Invalid CSR operation with t1
%t1 = "test.op"() : () -> !riscv.reg<t1>
%4 = riscv.csrrc %t1, 16, "r" : (!riscv.reg<t1>) -> !riscv.reg

// CHECK: Operation does not verify: When in 'readonly' mode, source must be register x0 (a.k.a. 'zero'), not 't1'

// -----

// Invalid CSR operation with wrong flag
%a2 = "test.op"() : () -> !riscv.reg<a2>
%5 = riscv.csrrw %a2, 16, "x" : (!riscv.reg<a2>) -> !riscv.reg

// CHECK: Expected 'w' flag, got 'x'

// -----

// Invalid CSR operation with wrong flag
%a3 = "test.op"() : () -> !riscv.reg<a3>
%6 = riscv.csrrs %a3, 16, "x" : (!riscv.reg<a3>) -> !riscv.reg

// CHECK: Expected 'r' flag, got 'x'
