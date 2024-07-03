// RUN: xdsl-opt --split-input-file --verify-diagnostics --parsing-diagnostics %s | filecheck %s

%i1 = "test.op"() : () -> !riscv.reg<a1>
%1 = riscv.li 1 : () -> !riscv.reg

%empty_0 = riscv_snitch.scfgw %i1, %1 : (!riscv.reg<a1>, !riscv.reg) -> !riscv.reg

// CHECK: Operation does not verify: scfgw rd must be ZERO, got !riscv.reg

// -----

%i1 = "test.op"() : () -> !riscv.reg<a1>
%1 = riscv.li 1 : () -> !riscv.reg
%wrong_0 = riscv_snitch.scfgw %i1, %1 : (!riscv.reg<a1>, !riscv.reg) -> !riscv.reg<t0>

// CHECK: Operation does not verify: scfgw rd must be ZERO, got !riscv.reg<t0>

// -----

%i1 = "test.op"() : () -> !riscv.reg<a1>
%empty_1 = riscv_snitch.scfgwi %i1, 1 : (!riscv.reg<a1>) -> !riscv.reg

// CHECK: Operation does not verify: scfgwi rd must be ZERO, got !riscv.reg

// -----

%i1 = "test.op"() : () -> !riscv.reg<a1>
%wrong_1 = riscv_snitch.scfgwi %i1, 1 : (!riscv.reg<a1>) -> !riscv.reg<t0>

// CHECK: Operation does not verify: scfgwi rd must be ZERO, got !riscv.reg<t0>

// -----

%i1 = "test.op"() : () -> !riscv.reg<a1>
%ok_imm = riscv.addi %i1, 1 : (!riscv.reg<a1>) -> !riscv.reg<t0>
%big_imm = riscv.addi %i1, 2048 : (!riscv.reg<a1>) -> !riscv.reg<t0>

// CHECK: Integer value 2048 is out of range for type si12 which supports values in the range [-2048, 2048)
