// RUN: xdsl-opt --split-input-file --verify-diagnostics %s | filecheck %s

// ParallelMov Operation

// Mismatched input and output counts
%0, %1, %2 = "test.op"() : () -> (!riscv.reg<s0>, !riscv.reg<s1>, !riscv.reg<s2>)
riscv.parallel_mov %0, %1, %2 : (!riscv.reg<s0>, !riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg<s3>, !riscv.reg<s4>)

// CHECK: incorrect length

// -----

// Moving from int register to float register
%0, %1 = "test.op"() : () -> (!riscv.reg<s0>, !riscv.reg<s1>)
riscv.parallel_mov %0, %1 : (!riscv.reg<s0>, !riscv.reg<s1>) -> (!riscv.freg<f0>, !riscv.freg<f1>)

// CHECK: %0, %1 = "test.op"() : () -> (!riscv.reg<s0>, !riscv.reg<s1>)
// CHECK: Input type must match output type.

// -----

// Duplicated output registers
%0, %1 = "test.op"() : () -> (!riscv.reg<s0>, !riscv.reg<s1>)
riscv.parallel_mov %0, %1 : (!riscv.reg<s0>, !riscv.reg<s1>) -> (!riscv.reg<s5>, !riscv.reg<s5>)

// CHECK: %0, %1 = "test.op"() : () -> (!riscv.reg<s0>, !riscv.reg<s1>)
// CHECK: Outputs must be unallocated or distinct.
