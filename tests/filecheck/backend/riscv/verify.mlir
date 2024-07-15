// RUN: xdsl-opt --split-input-file --verify-diagnostics --parsing-diagnostics %s | filecheck %s

%i1 = "test.op"() : () -> !riscv.reg<a1>
%ok_imm = riscv.addi %i1, 1 : (!riscv.reg<a1>) -> !riscv.reg<t0>
%big_imm = riscv.addi %i1, 2048 : (!riscv.reg<a1>) -> !riscv.reg<t0>

// CHECK: Integer value 2048 is out of range for type si12 which supports values in the range [-2048, 2048)
